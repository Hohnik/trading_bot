"""
RSI Bidirectional Strategy
A mean-reversion strategy that longs oversold and shorts overbought conditions.
"""

import pandas as pd
from typing import Optional, Dict, Any
from . import indicators


class RSIBidirectionalStrategy:
    """A strategy that trades both long and short based on RSI extremes, exiting at the mean."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.name = 'rsi_bidirectional'
        self.config = config or {}
        self.leverage = self.config.get('leverage', 2.0)
        self.base_position_size = self.config.get('position_size', 0.25)
        self.hold_max_bars = self.config.get('hold_max_bars', 48) # 12 days
        self.profit_target = self.config.get('profit_target', 0.12)
        self.stop_loss = self.config.get('stop_loss', 0.05)
        self.trailing_stop = self.config.get('trailing_stop', 0.07)
        self.rsi_period = self.config.get('rsi_period', 14)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.rsi_exit = self.config.get('rsi_exit', 50)
        self.rsi_confirm = self.config.get('rsi_confirmation', 3)

    def check_entry(self, data: pd.DataFrame, spy_data: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
        """Check for RSI oversold (long) or overbought (short) entry signals."""
        if len(data) < self.rsi_period + 2:
            return None

        rsi = indicators.calculate_rsi(data['Close'], period=self.rsi_period)
        if len(rsi) < 2: return None

        current_rsi, prev_rsi = rsi.iloc[-1], rsi.iloc[-2]
        if pd.isna(current_rsi) or pd.isna(prev_rsi): return None

        # LONG Entry: RSI is bouncing from oversold
        if prev_rsi <= self.rsi_oversold and current_rsi > prev_rsi and current_rsi <= (self.rsi_oversold + self.rsi_confirm):
            return {'type': 'long', 'price': data['Close'].iloc[-1], 'conviction': 8}

        # SHORT Entry: RSI is falling from overbought
        if prev_rsi >= self.rsi_overbought and current_rsi < prev_rsi and current_rsi >= (self.rsi_overbought - self.rsi_confirm):
            return {'type': 'short', 'price': data['Close'].iloc[-1], 'conviction': 8}
            
        return None

    def check_exit(self, position: Dict[str, Any], current_price: float) -> tuple[bool, Optional[str]]:
        """Check for price-based exit conditions for both long and short positions."""
        entry_price = position['entry_price']
        pos_type = position.get('type', 'long')
        
        price_change = (current_price / entry_price - 1) if pos_type == 'long' else (entry_price / current_price - 1)
        
        # Trailing Stop
        if pos_type == 'long':
            drawdown = (current_price / position.get('highest_price', entry_price) - 1)
            if drawdown <= -self.trailing_stop: return True, 'trailing_stop'
        else: # Short
            drawup = (current_price / position.get('lowest_price', current_price) - 1)
            if drawup >= self.trailing_stop: return True, 'trailing_stop'

        # Profit Target & Stop Loss
        if price_change >= self.profit_target: return True, 'profit_target'
        if price_change <= -self.stop_loss: return True, 'stop_loss'
        if position.get('bars_held', 0) >= self.hold_max_bars: return True, 'max_hold'
        
        return False, None

    def check_exit_signal(self, data: pd.DataFrame, position_type: str) -> bool:
        """Check if RSI has returned to the mean (50) to exit the position."""
        if len(data) < self.rsi_period + 2:
            return False

        rsi = indicators.calculate_rsi(data['Close'], period=self.rsi_period)
        if len(rsi) < 2: return False

        current_rsi, prev_rsi = rsi.iloc[-1], rsi.iloc[-2]
        if pd.isna(current_rsi) or pd.isna(prev_rsi): return False

        if position_type == 'long':
            return prev_rsi < self.rsi_exit and current_rsi >= self.rsi_exit
        else: # Short
            return prev_rsi > self.rsi_exit and current_rsi <= self.rsi_exit

    def get_position_size(self, conviction: int, balance: float, volatility: float = 0.02) -> float:
        """Return a position size for the bidirectional RSI strategy."""
        return self.base_position_size * balance
