import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def calculate_rsi_manual(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_trend_strength(prices, period=3):
    if len(prices) < period + 1:
        return 0
    trend = (
        (prices.iloc[-1] - prices.iloc[-1 - period]) / prices.iloc[-1 - period] * 100
    )
    return abs(trend)


def backtest(
    data,
    initial_balance=10000,
    leverage=5.0,
    risk_percent=0.01,
    profit_target_ratio=10.0,
    trading_fee_percent=0.001,
    rsi_period=14,
    rsi_confirmation_threshold=3,
    min_trend_strength=0.2,
):
    balance = initial_balance
    equity_curve = []

    position = None

    min_hold_time = 3
    volume_confirmation_factor = 1.1
    base_position_percent = 0.2
    min_position_percent = 0.1
    max_position_percent = 0.4

    last_trade_step = -min_hold_time

    data["rsi"] = calculate_rsi_manual(data["Close"], period=rsi_period)
    data["avg_volume"] = data["Volume"].rolling(window=20).mean()
    data["volatility"] = data["Close"].pct_change().rolling(window=10).std() * 100

    trades = []

    for i in range(21, len(data)):
        current_equity = balance
        if position:
            price_change = (data["Close"].iloc[i] - position["entry_price"]) / position[
                "entry_price"
            ]
            pnl_direction = 1 if position["type"] == "long" else -1
            unrealized_pnl = (
                price_change * position["entry_value"] * leverage * pnl_direction
            )
            current_equity += unrealized_pnl
        equity_curve.append(current_equity)

        if position:
            stop_loss_price_change = -risk_percent / (
                position["position_percent"] * leverage
            )
            take_profit_price_change = (
                risk_percent
                * profit_target_ratio
                / (position["position_percent"] * leverage)
            )
            price_change_percent = (
                data["Close"].iloc[i] - position["entry_price"]
            ) / position["entry_price"]
            pnl_direction = 1 if position["type"] == "long" else -1

            if (price_change_percent * pnl_direction) >= take_profit_price_change or (
                price_change_percent * pnl_direction
            ) <= stop_loss_price_change:
                entry_trade_value = position["entry_value"] * leverage
                exit_trade_value = (
                    data["Close"].iloc[i] / position["entry_price"]
                ) * entry_trade_value
                entry_fee = entry_trade_value * trading_fee_percent
                exit_fee = exit_trade_value * trading_fee_percent

                gross_pnl = (
                    (data["Close"].iloc[i] - position["entry_price"])
                    / position["entry_price"]
                    * position["entry_value"]
                    * leverage
                    * pnl_direction
                )
                net_pnl = gross_pnl - entry_fee - exit_fee

                balance += net_pnl
                trades.append(
                    {
                        "entry_date": position["entry_date"],
                        "exit_date": data.index[i],
                        "pnl": net_pnl,
                        "type": position["type"],
                    }
                )
                position = None
                last_trade_step = i
                continue

        if not position:
            if (i - last_trade_step) < min_hold_time:
                continue

            prices_slice = data["Close"].iloc[: i + 1]
            current_rsi = data["rsi"].iloc[i]
            previous_rsi = data["rsi"].iloc[i - 1]
            volatility = data["volatility"].iloc[i]

            if pd.isna(current_rsi) or pd.isna(previous_rsi) or pd.isna(volatility):
                continue

            position_percent = base_position_percent
            if current_rsi > 60:
                position_percent *= 1.1
            elif current_rsi < 40:
                position_percent *= 0.9

            trend_strength = calculate_trend_strength(prices_slice)
            if trend_strength > 1.0:
                position_percent *= 1.2
            elif trend_strength < 0.3:
                position_percent *= 0.8

            if volatility < 0.5:
                position_percent *= 1.05
            elif volatility > 1.5:
                position_percent *= 0.95

            position_percent = np.clip(
                position_percent, min_position_percent, max_position_percent
            )

            rsi_crossed_above = previous_rsi <= 50 and current_rsi > 50
            rsi_confirmation = current_rsi >= (50 + rsi_confirmation_threshold)
            volume_confirmation = (
                data["Volume"].iloc[i]
                > data["avg_volume"].iloc[i] * volume_confirmation_factor
            )
            trend_confirmation = trend_strength >= min_trend_strength
            price_momentum = (
                len(prices_slice) > 3 and prices_slice.iloc[-1] > prices_slice.iloc[-3]
            )

            if (
                rsi_crossed_above
                and rsi_confirmation
                and volume_confirmation
                and trend_confirmation
                and price_momentum
            ):
                entry_value = balance * position_percent
                position = {
                    "type": "long",
                    "entry_price": data["Close"].iloc[i],
                    "entry_date": data.index[i],
                    "entry_value": entry_value,
                    "position_percent": position_percent,
                }
                last_trade_step = i
                continue

            rsi_crossed_below = previous_rsi >= 50 and current_rsi < 50
            rsi_confirmation = current_rsi <= (50 - rsi_confirmation_threshold)
            price_momentum_sell = (
                len(prices_slice) > 3 and prices_slice.iloc[-1] < prices_slice.iloc[-3]
            )

            if (
                rsi_crossed_below
                and rsi_confirmation
                and volume_confirmation
                and trend_confirmation
                and price_momentum_sell
            ):
                entry_value = balance * position_percent
                position = {
                    "type": "short",
                    "entry_price": data["Close"].iloc[i],
                    "entry_date": data.index[i],
                    "entry_value": entry_value,
                    "position_percent": position_percent,
                }
                last_trade_step = i
                continue

    if position:
        pnl_direction = 1 if position["type"] == "long" else -1
        gross_pnl = (
            (data["Close"].iloc[-1] - position["entry_price"])
            / position["entry_price"]
            * position["entry_value"]
            * leverage
            * pnl_direction
        )
        balance += gross_pnl
        trades.append(
            {
                "entry_date": position["entry_date"],
                "exit_date": data.index[-1],
                "pnl": gross_pnl,
                "type": position["type"],
            }
        )

    equity_curve.append(balance)

    return equity_curve, trades


def main():
    os.makedirs("plots", exist_ok=True)

    tickers = [
        "AAPL",
        "GOOGL",
        "MSFT",
        "BTC-USD",  # Original
        "AMZN",
        "NVDA",
        "META",
        "TSLA",  # Tech
        "JPM",
        "V",
        "MA",  # Finance
        "JNJ",
        "UNH",  # Healthcare
        "WMT",
        "COST",
        "NKE",  # Consumer
        "SPY",
        "QQQ",
        "GLD",  # ETFs
        "ETH-USD",
        "XRP-USD",
        "ADA-USD",
        "DOGE-USD",
        "LTC-USD",  # Crypto
    ]

    # Hyperparameter tuning
    rsi_periods = [10, 14, 20]
    rsi_confirmations = [2, 3, 4]
    trend_strengths = [0.1, 0.2, 0.3]

    best_params = {}
    best_avg_return = -float("inf")

    for rsi_p in rsi_periods:
        for rsi_c in rsi_confirmations:
            for trend_s in trend_strengths:
                total_returns = []
                print(
                    f"Testing params: RSI Period={rsi_p}, RSI Confirm={rsi_c}, Trend Strength={trend_s}"
                )

                for ticker in tickers:
                    data = yf.download(
                        ticker,
                        start="2020-01-01",
                        end="2023-12-31",
                        interval="1d",
                        auto_adjust=True,
                        progress=False,
                    )
                    if data.empty:
                        continue
                    data.dropna(inplace=True)
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.droplevel(1)

                    equity_curve, trades = backtest(
                        data.copy(),
                        rsi_period=rsi_p,
                        rsi_confirmation_threshold=rsi_c,
                        min_trend_strength=trend_s,
                    )
                    if not equity_curve:
                        total_return = 0
                    else:
                        total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
                    total_returns.append(total_return)

                avg_return = np.mean(total_returns)
                print(f"  => Average Return: {avg_return:.2f}%")

                if avg_return > best_avg_return:
                    best_avg_return = avg_return
                    best_params = {
                        "rsi_period": rsi_p,
                        "rsi_confirmation_threshold": rsi_c,
                        "min_trend_strength": trend_s,
                    }

    print("\n--- Hyperparameter Tuning Complete ---")
    print(f"Best Average Return: {best_avg_return:.2f}%")
    print("Best Parameters:")
    print(best_params)

    print("\n--- Running Final Backtest with Best Parameters ---")
    for ticker in tickers:
        data = yf.download(
            ticker,
            start="2020-01-01",
            end="2023-12-31",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if data.empty:
            continue
        data.dropna(inplace=True)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        equity_curve, trades = backtest(data.copy(), **best_params)

        total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
        wins = sum(1 for t in trades if t["pnl"] > 0)
        win_rate = wins / len(trades) * 100 if trades else 0

        print(f"\n--- {ticker} Results ---")
        print(f"Total return: {total_return:.2f}%")
        print(f"Number of trades: {len(trades)}")
        print(f"Win rate: {win_rate:.2f}%")

        plt.figure(figsize=(12, 6))
        plt.plot(data.index[: len(equity_curve)], equity_curve)
        plt.title(f"Equity Curve for {ticker} (Optimized)")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.grid(True)
        plt.savefig(f"plots/{ticker}_equity_curve_optimized.png")
        print(f"Saved equity curve plot to plots/{ticker}_equity_curve_optimized.png")
        plt.close()


if __name__ == "__main__":
    main()

