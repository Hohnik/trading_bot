"""
MACD STRATEGY
Enters on MACD crossovers with trend confirmation.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from . import indicators
from .base_strategy import BaseStrategy


class MACDStrategy(BaseStrategy):
    """A MACD strategy that enters on strong crossovers with trend confirmation."""

    def __init__(self, config: Dict[str, Any] = None):
        self.name = 'macd'
        self.leverage = 1.5
        super().__init__(config)
        self.leverage = self.config.get('leverage', 1.5)
        self.base_position_size = self.config.get('position_size', 0.3)
        self.hold_max_bars = self.config.get('hold_max_bars', 40)
        self.profit_target = self.config.get('profit_target', 0.15)
        self.stop_loss = self.config.get('stop_loss', 0.05)
        self.trailing_stop = self.config.get('trailing_stop', 0.08)

    def check_entry(self, data: pd.DataFrame, spy_data: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
        """Check for MACD entry signal."""
        if len(data) < 50:
            return None

        close = data['Close']
        macd, signal_line, histogram = indicators.calculate_macd(close)

        ema_trend = close.ewm(span=50, adjust=False).mean()

        latest = {
            'close': close.iloc[-1],
            'macd': macd.iloc[-1],
            'signal': signal_line.iloc[-1],
            'histogram': histogram.iloc[-1],
            'prev_histogram': histogram.iloc[-2],
            'ema_trend': ema_trend.iloc[-1]
        }

        conviction = 0

        # Bullish crossover
        if (latest['macd'] > latest['signal'] and
            latest['prev_histogram'] < 0 and latest['histogram'] > 0):
            conviction += 2
            if latest['close'] > latest['ema_trend']:
                conviction += 1
            if abs(latest['histogram']) > 0.5:
                conviction += 1

            if conviction >= 3:
                return {
                    'type': 'long',
                    'price': latest['close'],
                    'conviction': conviction
                }

        # Bearish crossover
        if (latest['macd'] < latest['signal'] and
            latest['prev_histogram'] > 0 and latest['histogram'] < 0):
            conviction += 2
            if latest['close'] < latest['ema_trend']:
                conviction += 1
            if abs(latest['histogram']) > 0.5:
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
