"""
Common technical indicators used by trading strategies.
"""
import pandas as pd

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using exponential moving average for better responsiveness."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_relative_strength(asset_prices: pd.Series, spy_prices: pd.Series) -> pd.Series:
    """Calculate relative strength vs SPY."""
    common_index = asset_prices.index.intersection(spy_prices.index)
    asset_aligned = asset_prices.loc[common_index]
    spy_aligned = spy_prices.loc[common_index]
    
    asset_returns = asset_aligned.pct_change(20)
    spy_returns = spy_aligned.pct_change(20)
    
    relative_strength = asset_returns - spy_returns
    return relative_strength.reindex(asset_prices.index).fillna(0)

def detect_volatility_expansion(atr: pd.Series, period: int = 10) -> pd.Series:
    """Detect when volatility is expanding."""
    atr_ma = atr.rolling(window=period).mean()
    return atr > atr_ma * 1.3
