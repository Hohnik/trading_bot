# RSI Crossover Trading Strategy

## 🚀 Strategy Overview

This document outlines an RSI (Relative Strength Index) crossover trading strategy. The strategy aims to identify and capitalize on market momentum by entering trades when the RSI crosses the 50 midline and exiting on a reverse crossover. The strategy has been backtested and optimized on a diverse set of assets.

## 🎯 Core Strategy Logic

-   **Buy Signal**: The RSI crosses above the 50 level, with a confirmation of the momentum.
-   **Sell Signal (Exit Long)**: The RSI crosses below the 50 level, with a confirmation.
-   **Risk Management**: Each trade is managed with a predefined risk-per-trade percentage and a stop-loss/take-profit mechanism.

## ⚙️ Optimized & Fixed Parameters

The following parameters were found to be optimal during a hyperparameter tuning process that maximized the average return across a basket of 24 diverse assets (stocks, ETFs, and cryptocurrencies).

### Tuned Parameters
-   **RSI Period**: 10
-   **RSI Confirmation Threshold**: 4 points (i.e., RSI must be >= 54 for a buy or <= 46 for a sell)
-   **Minimum Trend Strength**: 0.2%

### Fixed Parameters
-   **Leverage**: 5.0x
-   **Risk Per Trade**: 1% of total portfolio balance
-   **Profit Target Ratio**: 10:1 (Take Profit is set at 10x the risk)
-   **Trading Fees**: 0.1% per side (0.2% round-trip)
-   **Base Position Size**: 20% of portfolio (with dynamic adjustments)

## 🔧 Key Implementation Code

The following Python functions are used for the core calculations in this strategy.

### RSI Calculation
```python
def calculate_rsi_manual(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
```

### Trend Strength Calculation
```python
def calculate_trend_strength(prices, period=3):
    if len(prices) < period + 1:
        return 0
    trend = (prices.iloc[-1] - prices.iloc[-1 - period]) / prices.iloc[-1 - period] * 100
    return abs(trend)
```

## 📊 Backtesting Results (Optimized)

The strategy was backtested from 2020-01-01 to 2023-12-31 using the optimized parameters. The results for each asset are as follows:

| Ticker  | Total Return | # of Trades | Win Rate |
|:--------|:-------------|:------------|:---------|
| AAPL    | 17.04%       | 15          | 26.67%   |
| GOOGL   | -42.47%      | 24          | 4.17%    |
| MSFT    | 28.76%       | 22          | 27.27%   |
| BTC-USD | 134.85%      | 35          | 37.14%   |
| AMZN    | -8.49%       | 16          | 12.50%   |
| NVDA    | -10.43%      | 21          | 19.05%   |
| META    | -2.35%       | 18          | 22.22%   |
| TSLA    | 27.41%       | 12          | 33.33%   |
| JPM     | 11.75%       | 17          | 23.53%   |
| V       | 8.77%        | 21          | 19.05%   |
| MA      | -9.91%       | 16          | 18.75%   |
| JNJ     | 24.30%       | 19          | 26.32%   |
| UNH     | -28.72%      | 26          | 15.38%   |
| WMT     | -33.36%      | 25          | 8.00%    |
| COST    | 23.18%       | 22          | 27.27%   |
| NKE     | -7.14%       | 21          | 23.81%   |
| SPY     | 26.18%       | 17          | 23.53%   |
| QQQ     | -5.84%       | 14          | 14.29%   |
| GLD     | -17.88%      | 19          | 10.53%   |
| ETH-USD | 86.32%       | 31          | 38.71%   |
| XRP-USD | -32.06%      | 28          | 17.86%   |
| ADA-USD | -35.40%      | 28          | 21.43%   |
| DOGE-USD| -16.25%      | 21          | 23.81%   |
| LTC-USD | 114.00%      | 20          | 40.00%   |
