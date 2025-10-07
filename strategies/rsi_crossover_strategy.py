"""
RSI Crossover Strategy
Trades RSI crosses of the 50-level centerline as a momentum signal.
"""

import pandas as pd
from typing import Optional, Dict, Any
from . import indicators


class RSICrossoverStrategy:
    """A strategy that buys on RSI crossing above 50 and sells on crossing below 50."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.name = 'rsi_crossover'
        self.config = config or {}
        self.leverage = self.config.get('leverage', 1.5)
        self.base_position_size = self.config.get('position_size', 0.20)
        self.hold_max_bars = self.config.get('hold_max_bars', 24) # 6 days
        self.profit_target = self.config.get('profit_target', 0.15)
        self.stop_loss = self.config.get('stop_loss', 0.05)
        self.trailing_stop = self.config.get('trailing_stop', 0.08)
        self.rsi_period = self.config.get('rsi_period', 10)
        self.rsi_confirm = self.config.get('rsi_confirmation_threshold', 4)

    def check_entry(self, data: pd.DataFrame, spy_data: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
        """Check for an RSI crossover entry signal."""
        if len(data) < self.rsi_period + 2:
            return None

        rsi = indicators.calculate_rsi(data['Close'], period=self.rsi_period)
        if len(rsi) < 2: return None

        current_rsi, prev_rsi = rsi.iloc[-1], rsi.iloc[-2]
        if pd.isna(current_rsi) or pd.isna(prev_rsi): return None

        if prev_rsi <= 50 and current_rsi > (50 + self.rsi_confirm):
            return {
                'type': 'long',
                'price': data['Close'].iloc[-1],
                'conviction': 7
            }
        return None

    def check_exit(self, position: Dict[str, Any], current_price: float) -> tuple[bool, Optional[str]]:
        """Check for price-based exit conditions."""
        entry_price = position['entry_price']
        highest_price = position.get('highest_price', entry_price)
        price_change = (current_price / entry_price - 1)
        drawdown = (current_price / highest_price - 1)

        if price_change >= self.profit_target: return True, 'profit_target'
        if price_change <= -self.stop_loss: return True, 'stop_loss'
        if drawdown <= -self.trailing_stop: return True, 'trailing_stop'
        if position.get('bars_held', 0) >= self.hold_max_bars: return True, 'max_hold'
        
        return False, None

    def check_exit_signal(self, data: pd.DataFrame, position_type: str) -> bool:
        """Check for an RSI crossover exit signal."""
        if len(data) < self.rsi_period + 2:
            return False

        rsi = indicators.calculate_rsi(data['Close'], period=self.rsi_period)
        if len(rsi) < 2: return False

        current_rsi, prev_rsi = rsi.iloc[-1], rsi.iloc[-2]
        if pd.isna(current_rsi) or pd.isna(prev_rsi): return False

        return prev_rsi >= 50 and current_rsi < (50 - self.rsi_confirm)

    def get_position_size(self, conviction: int, balance: float, volatility: float = 0.02) -> float:
        """Return a position size for the RSI crossover strategy."""
        return self.base_position_size * balance
