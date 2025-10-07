"""
RSI Threshold Strategy
A classic mean-reversion strategy that buys oversold conditions.
"""

import pandas as pd
from typing import Optional, Dict, Any
from . import indicators


class RSIThresholdStrategy:
    """A strategy that buys when RSI is oversold (<30) and exits when overbought (>70)."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.name = 'rsi_threshold'
        self.config = config or {}
        self.leverage = self.config.get('leverage', 1.5)
        self.base_position_size = self.config.get('position_size', 0.25)
        self.hold_max_bars = self.config.get('hold_max_bars', 48) # 12 days
        self.profit_target = self.config.get('profit_target', 0.10)
        self.stop_loss = self.config.get('stop_loss', 0.04)
        self.trailing_stop = self.config.get('trailing_stop', 0.06)
        self.rsi_period = self.config.get('rsi_period', 14)
        self.rsi_buy = self.config.get('rsi_buy_threshold', 30)
        self.rsi_sell = self.config.get('rsi_sell_threshold', 70)
        self.rsi_confirm = self.config.get('rsi_confirmation', 3)

    def check_entry(self, data: pd.DataFrame, spy_data: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
        """Check for an RSI oversold entry signal."""
        if len(data) < self.rsi_period + 2:
            return None

        rsi = indicators.calculate_rsi(data['Close'], period=self.rsi_period)
        if len(rsi) < 2: return None

        current_rsi, prev_rsi = rsi.iloc[-1], rsi.iloc[-2]
        if pd.isna(current_rsi) or pd.isna(prev_rsi): return None

        if prev_rsi <= self.rsi_buy and current_rsi > prev_rsi and current_rsi <= (self.rsi_buy + self.rsi_confirm):
            return {
                'type': 'long',
                'price': data['Close'].iloc[-1],
                'conviction': 8
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
        """Check for an RSI overbought exit signal."""
        if len(data) < self.rsi_period + 2:
            return False

        rsi = indicators.calculate_rsi(data['Close'], period=self.rsi_period)
        if len(rsi) < 2: return False

        current_rsi, prev_rsi = rsi.iloc[-1], rsi.iloc[-2]
        if pd.isna(current_rsi) or pd.isna(prev_rsi): return False

        return prev_rsi < self.rsi_sell and current_rsi >= self.rsi_sell

    def get_position_size(self, conviction: int, balance: float, volatility: float = 0.02) -> float:
        """Return a position size for the RSI threshold strategy."""
        return self.base_position_size * balance
