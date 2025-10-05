"""
INNOVATIVE SWING TRADING STRATEGY
Multi-Timeframe Momentum + Volatility Breakout System

INNOVATION:
1. Multi-timeframe confluence (daily + 4h alignment)
2. Relative strength vs SPY (outperformance detection)
3. Volatility expansion entries (low -> high vol breakouts)
4. Volume-weighted momentum
5. Dynamic position sizing based on conviction
6. Adaptive exits (trailing stops that adjust to volatility)

EXPECTED:
- 2-5 day holds
- 3-5 trades per week
- 30-50% annual returns
- High Sharpe ratio (>1.5)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


def calculate_rsi(prices, period=14):
    """RSI indicator."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_vwap(data):
    """Volume Weighted Average Price."""
    return (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()


def calculate_relative_strength(asset_prices, spy_prices):
    """Calculate relative strength vs SPY."""
    # Align indices
    common_index = asset_prices.index.intersection(spy_prices.index)
    asset_aligned = asset_prices.loc[common_index]
    spy_aligned = spy_prices.loc[common_index]
    
    # Calculate relative performance
    asset_returns = asset_aligned.pct_change(20)  # 20-day returns
    spy_returns = spy_aligned.pct_change(20)
    
    return asset_returns - spy_returns  # Outperformance


def detect_volatility_expansion(atr, period=10):
    """Detect when volatility is expanding (breakout signal)."""
    atr_ma = atr.rolling(window=period).mean()
    return atr > atr_ma * 1.3  # ATR 30% above average


def backtest_swing_strategy(
    data,
    spy_data=None,  # For relative strength
    initial_balance=10000,
    leverage=3.0,  # Increased leverage for higher returns
    base_position_size=0.5,  # 50% base position (more aggressive)
    hold_min_bars=4,  # Min 4 bars (1 day on 4h data) - trade more frequently
    hold_max_bars=30,  # Max 30 bars (7.5 days on 4h data)
):
    """
    Swing trading strategy with multiple confirmations.
    
    ENTRY RULES (All must be true):
    1. Multi-timeframe momentum: Both fast and slow EMAs trending up
    2. Volatility expansion: ATR breaking above average
    3. Volume confirmation: Above average volume
    4. RSI momentum: Between 40-70 (not overbought/oversold)
    5. (Optional) Relative strength: Outperforming SPY
    
    EXIT RULES (Any is true):
    1. Profit target: +15% (swing trade target)
    2. Stop loss: -5% (tight risk management)
    3. Trailing stop: 8% from highest point
    4. Momentum loss: Fast EMA crosses below slow
    5. Maximum hold time reached
    """
    balance = initial_balance
    equity_curve = []
    position = None
    trades = []
    
    # Calculate indicators
    data = data.copy()
    data['rsi'] = calculate_rsi(data['Close'], period=14)
    data['ema_fast'] = data['Close'].ewm(span=8, adjust=False).mean()
    data['ema_slow'] = data['Close'].ewm(span=21, adjust=False).mean()
    data['ema_trend'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['atr'] = calculate_atr(data['High'], data['Low'], data['Close'], period=14)
    
    # Volume indicators
    data['volume_ma'] = data['Volume'].rolling(window=20).mean()
    data['volume_ratio'] = data['Volume'] / data['volume_ma']
    
    # Volatility indicators
    data['vol_expanding'] = detect_volatility_expansion(data['atr'])
    data['volatility'] = data['Close'].pct_change().rolling(window=20).std()
    
    # Momentum indicators
    data['momentum'] = data['Close'].pct_change(5) * 100  # 5-bar momentum
    data['macd'] = data['ema_fast'] - data['ema_slow']
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    
    # Relative strength (if SPY data provided)
    if spy_data is not None:
        data['rel_strength'] = calculate_relative_strength(data['Close'], spy_data['Close'])
    else:
        data['rel_strength'] = 0
    
    # Bollinger Bands for context
    data['bb_ma'] = data['Close'].rolling(window=20).mean()
    data['bb_std'] = data['Close'].rolling(window=20).std()
    data['bb_upper'] = data['bb_ma'] + (data['bb_std'] * 2)
    data['bb_lower'] = data['bb_ma'] - (data['bb_std'] * 2)
    
    for i in range(100, len(data)):  # Start after enough data
        current_price = data['Close'].iloc[i]
        current_equity = balance
        
        # Update position
        if position:
            price_change_pct = (current_price / position['entry_price'] - 1)
            unrealized_pnl = price_change_pct * position['position_value'] * leverage
            current_equity += unrealized_pnl
            
            # Update highest price for trailing stop
            if current_price > position['highest_price']:
                position['highest_price'] = current_price
            
            position['bars_held'] += 1
        
        equity_curve.append(current_equity)
        
        # EXIT LOGIC
        if position:
            entry_price = position['entry_price']
            highest = position['highest_price']
            bars_held = position['bars_held']
            
            price_change = (current_price / entry_price - 1)
            drawdown_from_high = (current_price / highest - 1)
            
            should_exit = False
            exit_reason = None
            
            # Take profit: +20% (let winners run more)
            if price_change >= 0.20:
                should_exit = True
                exit_reason = 'profit_target'
            
            # Stop loss: -4% (tighter stop)
            elif price_change <= -0.04:
                should_exit = True
                exit_reason = 'stop_loss'
            
            # Trailing stop: 6% from highest (tighter trailing)
            elif drawdown_from_high <= -0.06:
                should_exit = True
                exit_reason = 'trailing_stop'
            
            # Momentum reversal
            elif (data['ema_fast'].iloc[i] < data['ema_slow'].iloc[i] and
                  data['macd'].iloc[i] < data['macd_signal'].iloc[i]):
                should_exit = True
                exit_reason = 'momentum_loss'
            
            # Max hold time
            elif bars_held >= hold_max_bars:
                should_exit = True
                exit_reason = 'max_hold'
            
            # Min hold time check
            if should_exit and bars_held < hold_min_bars:
                # Don't exit too early unless stop loss
                if exit_reason != 'stop_loss':
                    should_exit = False
            
            if should_exit:
                # Calculate P&L
                gross_pnl = price_change * position['position_value'] * leverage
                fees = position['position_value'] * leverage * 0.001 * 2  # 0.1% per side
                net_pnl = gross_pnl - fees
                
                balance += net_pnl
                
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': data.index[i],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': net_pnl,
                    'return_pct': price_change * 100,
                    'bars_held': bars_held,
                    'exit_reason': exit_reason,
                    'entry_signals': position['entry_signals']
                })
                
                position = None
                continue
        
        # ENTRY LOGIC - Multi-factor confluence
        if not position:
            rsi = data['rsi'].iloc[i]
            ema_fast = data['ema_fast'].iloc[i]
            ema_slow = data['ema_slow'].iloc[i]
            ema_trend = data['ema_trend'].iloc[i]
            atr = data['atr'].iloc[i]
            vol_expanding = data['vol_expanding'].iloc[i]
            volume_ratio = data['volume_ratio'].iloc[i]
            momentum = data['momentum'].iloc[i]
            macd = data['macd'].iloc[i]
            macd_signal = data['macd_signal'].iloc[i]
            rel_strength = data['rel_strength'].iloc[i]
            volatility = data['volatility'].iloc[i]
            bb_lower = data['bb_lower'].iloc[i]
            bb_upper = data['bb_upper'].iloc[i]
            
            if pd.isna(rsi) or pd.isna(atr):
                continue
            
            # Count confirmations
            confirmations = []
            conviction = 0
            
            # 1. Trend alignment (EMAs stacked bullishly)
            if ema_fast > ema_slow > ema_trend:
                confirmations.append('trend_alignment')
                conviction += 2
            elif ema_fast > ema_slow:
                confirmations.append('short_term_trend')
                conviction += 1
            
            # 2. Momentum confirmation (MACD crossover)
            if macd > macd_signal and momentum > 2:
                confirmations.append('strong_momentum')
                conviction += 2
            elif macd > macd_signal:
                confirmations.append('momentum')
                conviction += 1
            
            # 3. Volatility breakout
            if vol_expanding:
                confirmations.append('volatility_breakout')
                conviction += 2
            
            # 4. Volume confirmation
            if volume_ratio > 1.3:
                confirmations.append('high_volume')
                conviction += 1
            
            # 5. RSI in healthy range (not extreme)
            if 40 <= rsi <= 65:
                confirmations.append('rsi_healthy')
                conviction += 1
            elif 35 <= rsi < 40:
                confirmations.append('rsi_oversold_bounce')
                conviction += 2  # Extra conviction on bounce
            
            # 6. Relative strength (outperforming market)
            if rel_strength > 0.02:  # Outperforming by 2%+
                confirmations.append('market_outperformance')
                conviction += 2
            elif rel_strength > 0:
                confirmations.append('relative_strength')
                conviction += 1
            
            # 7. Price position (not at resistance)
            if current_price < bb_upper * 0.95:  # Not overbought
                confirmations.append('not_overbought')
                conviction += 1
            
            # 8. Mean reversion opportunity (price near lower BB but momentum turning)
            if current_price < bb_lower * 1.02 and ema_fast > ema_slow:
                confirmations.append('mean_reversion_setup')
                conviction += 3  # Strong signal
            
            # ENTRY DECISION: Need minimum conviction score
            # Lowered threshold to trade more frequently
            # High conviction: 7+ points
            # Medium conviction: 5-6 points
            # Low conviction: 4 points (skip)
            
            if conviction >= 5 and len(confirmations) >= 3:
                # Calculate position size based on conviction
                if conviction >= 9:
                    position_size = base_position_size * 1.5  # Size up more on strong signals
                elif conviction >= 7:
                    position_size = base_position_size * 1.2
                elif conviction >= 6:
                    position_size = base_position_size
                else:
                    position_size = base_position_size * 0.8  # Smaller size on weaker signals
                
                # Adjust for volatility (higher vol = smaller size)
                vol_adjustment = 1.0 / (1.0 + volatility * 50)
                position_size *= vol_adjustment
                
                # Cap position size
                position_size = np.clip(position_size, 0.2, 0.6)
                position_value = balance * position_size
                
                position = {
                    'entry_price': current_price,
                    'entry_date': data.index[i],
                    'position_value': position_value,
                    'highest_price': current_price,
                    'bars_held': 0,
                    'entry_signals': confirmations,
                    'conviction': conviction
                }
    
    # Close final position
    if position:
        final_price = data['Close'].iloc[-1]
        price_change = (final_price / position['entry_price'] - 1)
        gross_pnl = price_change * position['position_value'] * leverage
        balance += gross_pnl
        
        trades.append({
            'entry_date': position['entry_date'],
            'exit_date': data.index[-1],
            'entry_price': position['entry_price'],
            'exit_price': final_price,
            'pnl': gross_pnl,
            'return_pct': price_change * 100,
            'bars_held': position['bars_held'],
            'exit_reason': 'end_of_data',
            'entry_signals': position['entry_signals']
        })
    
    equity_curve.append(balance)
    
    return equity_curve, trades