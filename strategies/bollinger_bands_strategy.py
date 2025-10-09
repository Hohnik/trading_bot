"""
BOLLINGER BANDS STRATEGY
Enters on mean reversion signals when price touches bands.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from . import indicators
from .base_strategy import BaseStrategy


class BollingerBandsStrategy(BaseStrategy):
    """A Bollinger Bands strategy that enters on mean reversion."""

    def __init__(self, config: Dict[str, Any] = None):
        self.name = 'bollinger'
        self.leverage = 1.5
        super().__init__(config)
        self.leverage = self.config.get('leverage', 1.5)
        self.base_position_size = self.config.get('position_size', 0.3)
        self.hold_max_bars = self.config.get('hold_max_bars', 20)
        self.profit_target = self.config.get('profit_target', 0.08)
        self.stop_loss = self.config.get('stop_loss', 0.04)
        self.trailing_stop = self.config.get('trailing_stop', 0.06)

    def check_entry(self, data: pd.DataFrame, spy_data: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
        """Check for Bollinger Bands entry signal."""
        if len(data) < 50:
            return None

        close = data['Close']
        upper, middle, lower = indicators.calculate_bollinger_bands(close)
        rsi = indicators.calculate_rsi(close)

        latest = {
            'close': close.iloc[-1],
            'upper': upper.iloc[-1],
            'middle': middle.iloc[-1],
            'lower': lower.iloc[-1],
            'rsi': rsi.iloc[-1]
        }

        bandwidth = (latest['upper'] - latest['lower']) / latest['middle']

        conviction = 0

        # Oversold - buy signal
        if latest['close'] <= latest['lower']:
            conviction += 2
            if latest['rsi'] < 35:
                conviction += 1
            if bandwidth > 0.08:
                conviction += 1

            if conviction >= 3:
                return {
                    'type': 'long',
                    'price': latest['close'],
                    'conviction': conviction
                }

        # Overbought - sell signal
        if latest['close'] >= latest['upper']:
            conviction += 2
            if latest['rsi'] > 65:
                conviction += 1
            if bandwidth > 0.08:
                conviction += 1

            if conviction >= 3:
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
        size = self.base_position_size * (1 + (conviction - 3) * 0.1)
        size *= (1.0 / (1.0 + volatility * 50))
        return np.clip(size, 0.1, 0.6) * balance
