"""
SWING TRADING STRATEGY
A multi-factor model that looks for momentum, trend, and volume alignment.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from . import indicators
from .base_strategy import BaseStrategy

class SwingStrategy(BaseStrategy):
    """A swing strategy that enters on strong technical confluence."""

    def __init__(self, config: Dict[str, Any] = None):
        self.name = 'swing'
        self.leverage = 2.0
        super().__init__(config)
        self.leverage = self.config.get('leverage', 2.0)
        self.base_position_size = self.config.get('position_size', 0.4)
        self.hold_max_bars = self.config.get('hold_max_bars', 30) # ~7.5 days
        self.profit_target = self.config.get('profit_target', 0.20)
        self.stop_loss = self.config.get('stop_loss', 0.04)
        self.trailing_stop = self.config.get('trailing_stop', 0.06)

    def check_entry(self, data: pd.DataFrame, spy_data: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
        """Check for a swing trade entry signal based on a confluence of indicators."""
        if len(data) < 50:
            return None

        # --- Calculate Indicators ---
        close = data['Close']
        ema_fast = close.ewm(span=8, adjust=False).mean()
        ema_slow = close.ewm(span=21, adjust=False).mean()
        ema_trend = close.ewm(span=50, adjust=False).mean()
        rsi = indicators.calculate_rsi(close)
        vol_ratio = data['Volume'] / data['Volume'].rolling(window=20).mean()
        rel_strength = indicators.calculate_relative_strength(close, spy_data['Close']) if spy_data is not None else pd.Series(0, index=close.index)

        # --- Latest Values ---
        latest = {
            'close': close.iloc[-1],
            'ema_fast': ema_fast.iloc[-1],
            'ema_slow': ema_slow.iloc[-1],
            'ema_trend': ema_trend.iloc[-1],
            'rsi': rsi.iloc[-1],
            'vol_ratio': vol_ratio.iloc[-1],
            'rs': rel_strength.iloc[-1]
        }

        # --- Entry Conditions (Confluence Model) ---
        conviction = 0
        if latest['ema_fast'] > latest['ema_slow'] > latest['ema_trend']: conviction += 2 # Strong Trend
        if 45 < latest['rsi'] < 70: conviction += 1                                  # Healthy Momentum
        if latest['vol_ratio'] > 1.3: conviction += 1                                # Volume Confirmation
        if latest['rs'] > 0.01: conviction += 1                                      # Market Outperformance

        if conviction >= 4:
            return {
                'type': 'long',
                'price': latest['close'],
                'conviction': conviction
            }
        return None

    def check_exit(self, position: Dict[str, Any], current_price: float) -> tuple[bool, Optional[str]]:
        """Check for a swing trade exit signal."""
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
        """Calculate position size based on conviction and volatility."""
        size = self.base_position_size * (1 + (conviction - 4) * 0.1) # Adjust size based on conviction
        size *= (1.0 / (1.0 + volatility * 50)) # Reduce size in high volatility
        return np.clip(size, 0.1, 0.7) * balance