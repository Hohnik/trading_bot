"""
DUAL THRUST STRATEGY
Range breakout strategy based on previous bar's range.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from .base_strategy import BaseStrategy


class DualThrustStrategy(BaseStrategy):
    """A dual thrust strategy that enters on range breakouts."""

    def __init__(self, config: Dict[str, Any] = None):
        self.name = 'dual_thrust'
        self.leverage = 1.5
        super().__init__(config)
        self.leverage = self.config.get('leverage', 1.5)
        self.base_position_size = self.config.get('position_size', 0.35)
        self.hold_max_bars = self.config.get('hold_max_bars', 15)
        self.profit_target = self.config.get('profit_target', 0.10)
        self.stop_loss = self.config.get('stop_loss', 0.04)
        self.trailing_stop = self.config.get('trailing_stop', 0.06)
        self.k1 = self.config.get('k1', 0.5)
        self.k2 = self.config.get('k2', 0.5)

    def check_entry(self, data: pd.DataFrame, spy_data: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
        """Check for dual thrust entry signal."""
        if len(data) < 10:
            return None

        latest = {
            'open': data['Open'].iloc[-1],
            'close': data['Close'].iloc[-1],
            'high': data['High'].iloc[-1],
            'low': data['Low'].iloc[-1],
            'prev_high': data['High'].iloc[-2],
            'prev_low': data['Low'].iloc[-2],
            'prev_close': data['Close'].iloc[-2]
        }

        range_val = max(
            latest['prev_high'] - latest['prev_close'],
            latest['prev_close'] - latest['prev_low']
        )

        buy_threshold = latest['open'] + self.k1 * range_val
        sell_threshold = latest['open'] - self.k2 * range_val

        conviction = 3

        # Breakout above threshold
        if latest['close'] > buy_threshold:
            return {
                'type': 'long',
                'price': latest['close'],
                'conviction': conviction
            }

        # Breakdown below threshold
        if latest['close'] < sell_threshold:
            return {
                'type': 'short',
                'price': latest['close'],
                'conviction': conviction
            }

        return None

    def check_exit(self, position: Dict[str, Any], current_price: float) -> tuple[bool, Optional[str]]:
        """Check for exit signal."""
        entry_price = position['entry_price']
        highest_price = position.get('highest_price', entry_price)
        lowest_price = position.get('lowest_price', entry_price)
        pos_type = position['type']

        if pos_type == 'long':
            price_change = (current_price / entry_price - 1)
            drawdown = (current_price / highest_price - 1)

            if price_change >= self.profit_target: return True, 'profit_target'
            if price_change <= -self.stop_loss: return True, 'stop_loss'
            if drawdown <= -self.trailing_stop: return True, 'trailing_stop'
        else:  # short
            price_change = (entry_price / current_price - 1)
            drawdown = (lowest_price / current_price - 1)

            if price_change >= self.profit_target: return True, 'profit_target'
            if price_change <= -self.stop_loss: return True, 'stop_loss'
            if drawdown <= -self.trailing_stop: return True, 'trailing_stop'

        if position.get('bars_held', 0) >= self.hold_max_bars: return True, 'max_hold'

        return False, None

    def get_position_size(self, conviction: int, balance: float, volatility: float = 0.02) -> float:
        """Calculate position size."""
        size = self.base_position_size
        size *= (1.0 / (1.0 + volatility * 50))
        return np.clip(size, 0.1, 0.6) * balance
