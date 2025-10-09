"""
Base class for all trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class that all trading strategies must inherit from."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the strategy.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._validate_required_attributes()

    def _validate_required_attributes(self):
        """Validate that required attributes are set."""
        if not hasattr(self, 'name') or not self.name:
            raise ValueError(f"{self.__class__.__name__} must define 'name' attribute")
        if not hasattr(self, 'leverage'):
            raise ValueError(f"{self.__class__.__name__} must define 'leverage' attribute")

    @abstractmethod
    def check_entry(self, data: pd.DataFrame, spy_data: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
        """
        Check for entry signals.

        Args:
            data: Price data DataFrame with OHLCV columns
            spy_data: Optional SPY data for relative strength calculations

        Returns:
            Signal dictionary with keys: 'type' ('long'|'short'), 'price', 'conviction'
            Returns None if no signal
        """
        pass

    @abstractmethod
    def check_exit(self, position: Dict[str, Any], current_price: float) -> tuple[bool, Optional[str]]:
        """
        Check for exit signals.

        Args:
            position: Current position dictionary with entry_price, highest_price, bars_held, etc.
            current_price: Current asset price

        Returns:
            Tuple of (should_exit: bool, reason: Optional[str])
        """
        pass

    @abstractmethod
    def get_position_size(self, conviction: int, balance: float, volatility: float = 0.02) -> float:
        """
        Calculate position size.

        Args:
            conviction: Signal conviction level (typically 1-10)
            balance: Available capital
            volatility: Asset volatility (default 0.02)

        Returns:
            Position size in dollars
        """
        pass
