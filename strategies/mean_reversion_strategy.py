"""
MEAN REVERSION STRATEGY
Trades oversold bounces using RSI and Bollinger Bands.
"""

import pandas as pd
from typing import Optional, Dict, Any
from . import indicators


class MeanReversionStrategy:
    """A strategy that buys oversold assets near their Bollinger Band support."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.name = 'mean_reversion'
        self.config = config or {}
        self.leverage = self.config.get('leverage', 1.5)
        self.base_position_size = self.config.get('position_size', 0.3)
        self.hold_max_bars = self.config.get('hold_max_bars', 20) # 5 days
        self.profit_target = self.config.get('profit_target', 0.08)
        self.stop_loss = self.config.get('stop_loss', 0.03)
        self.trailing_stop = self.config.get('trailing_stop', 0.04)

    def check_entry(self, data: pd.DataFrame, spy_data: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
        """Check for a mean reversion entry signal."""
        if len(data) < 21:
            return None

        # --- Calculate Indicators ---
        close = data['Close']
        rsi = indicators.calculate_rsi(close)
        bb_upper, _, bb_lower = indicators.calculate_bollinger_bands(close)
        bb_position = (close - bb_lower) / (bb_upper - bb_lower)

        # --- Latest Values ---
        latest = {
            'close': close.iloc[-1],
            'rsi': rsi.iloc[-1],
            'bb_pos': bb_position.iloc[-1],
            'bb_lower': bb_lower.iloc[-1]
        }

        # --- Entry Conditions ---
        is_oversold = latest['rsi'] < 35
        near_support = latest['bb_pos'] < 0.2
        is_bouncing = latest['close'] > latest['bb_lower']

        if is_oversold and near_support and is_bouncing:
            return {
                'type': 'long',
                'price': latest['close'],
                'conviction': 6
            }
        return None

    def check_exit(self, position: Dict[str, Any], current_price: float) -> tuple[bool, Optional[str]]:
        """Check for a mean reversion exit signal."""
        entry_price = position['entry_price']
        highest_price = position.get('highest_price', entry_price)
        price_change = (current_price / entry_price - 1)
        drawdown = (current_price / highest_price - 1)

        if price_change >= self.profit_target: return True, 'profit_target'
        if price_change <= -self.stop_loss: return True, 'stop_loss'
        if drawdown <= -self.trailing_stop: return True, 'trailing_stop'
        if position.get('bars_held', 0) >= self.hold_max_bars: return True, 'max_hold'
        
        return False, None

    def get_position_size(self, conviction: int, balance: float, volatility: float = 0.02) -> float:
        """Return a position size for the mean reversion strategy."""
        return self.base_position_size * balance
