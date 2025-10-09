"""
MOMENTUM BREAKOUT STRATEGY
Catches strong momentum moves with volume confirmation.
"""
import pandas as pd
from typing import Optional, Dict, Any
from . import indicators
from .base_strategy import BaseStrategy

class MomentumBreakoutStrategy(BaseStrategy):
    """A strategy that enters on breakouts with high volume and momentum."""

    def __init__(self, config: Dict[str, Any] = None):
        self.name = 'momentum'
        self.leverage = 1.5
        super().__init__(config)
        self.leverage = self.config.get('leverage', 1.5)
        self.base_position_size = self.config.get('position_size', 0.3)
        self.hold_max_bars = self.config.get('hold_max_bars', 12) # 3 days
        self.profit_target = self.config.get('profit_target', 0.10)
        self.stop_loss = self.config.get('stop_loss', 0.03)
        self.trailing_stop = self.config.get('trailing_stop', 0.05)

    def check_entry(self, data: pd.DataFrame, spy_data: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
        """Check for a momentum breakout entry signal."""
        if len(data) < 21:
            return None

        # --- Calculate Indicators ---
        close = data['Close']
        rsi = indicators.calculate_rsi(close)
        high_20 = data['High'].rolling(20).max().shift(1) # Previous 20-bar high
        vol_ratio = data['Volume'] / data['Volume'].rolling(20).mean()
        momentum_3 = close.pct_change(3)

        # --- Latest Values ---
        latest = {
            'close': close.iloc[-1],
            'rsi': rsi.iloc[-1],
            'high_20': high_20.iloc[-1],
            'vol_ratio': vol_ratio.iloc[-1],
            'momentum_3': momentum_3.iloc[-1]
        }

        # --- Entry Conditions ---
        is_breakout = latest['close'] > latest['high_20']
        has_volume = latest['vol_ratio'] > 1.5
        has_momentum = latest['momentum_3'] > 0.03
        in_healthy_range = 50 < latest['rsi'] < 80

        if is_breakout and has_volume and has_momentum and in_healthy_range:
            return {
                'type': 'long',
                'price': latest['close'],
                'conviction': 7
            }
        return None

    def check_exit(self, position: Dict[str, Any], current_price: float) -> tuple[bool, Optional[str]]:
        """Check for a momentum trade exit signal."""
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
        """Return a position size for the momentum strategy."""
        return self.base_position_size * balance
