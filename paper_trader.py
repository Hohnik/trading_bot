"""
Multi-Strategy Paper Trader - A lean, multi-strategy paper trading system.
"""

import yfinance as yf
import pandas as pd
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import strategies
from strategies.swing_strategy import SwingStrategy
from strategies.momentum_strategy import MomentumBreakoutStrategy
from strategies.rsi_crossover_strategy import RSICrossoverStrategy
from strategies.rsi_threshold_strategy import RSIThresholdStrategy
from strategies.rsi_bidirectional_strategy import RSIBidirectionalStrategy


class PaperTrader:
    """A lean, multi-strategy paper trading system."""

    def __init__(self, strategies: List[Any], balance_per_strategy: float = 100):
        self.strategies = {s.name: s for s in strategies}
        self.initial_balance = balance_per_strategy * len(strategies)
        self.balances = {s.name: balance_per_strategy for s in strategies}
        self.positions = {s.name: {} for s in strategies}
        self.trade_history = {s.name: [] for s in strategies}

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            handlers=[logging.FileHandler("paper_trader.log"), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"🚀 Trader initialized with {len(strategies)} strategies (${balance_per_strategy:,.0f} each)."
        )

    def get_market_data(
        self, ticker: str, interval: str = "1h", period: str = "5d"
    ) -> Optional[pd.DataFrame]:
        """Fetches and prepares market data, resampling to a 4h timeframe."""
        try:
            data = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            if data.empty or len(data) < 4:
                return None
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            return (
                data.resample("4h")
                .agg(
                    {
                        "Open": "first",
                        "High": "max",
                        "Low": "min",
                        "Close": "last",
                        "Volume": "sum",
                    }
                )
                .dropna()
            )
        except Exception as e:
            self.logger.error(f"Error fetching {ticker}: {e}")
            return None

    def run(self, tickers: List[str], check_interval: int = 3600):
        """Main trading loop."""
        self.logger.info(
            f"Watching: {', '.join(tickers)} | Interval: {check_interval / 60:.0f} min\n"
        )
        last_check = 0
        try:
            while True:
                if time.time() - last_check < check_interval:
                    time.sleep(60)
                    continue

                self.logger.info(
                    f"--- Market Check @ {datetime.now().strftime('%H:%M:%S')} ---"
                )
                spy_data = self.get_market_data("SPY")
                market_data = {t: self.get_market_data(t) for t in tickers}
                current_prices = {
                    t: d["Close"].iloc[-1]
                    for t, d in market_data.items()
                    if d is not None
                }

                for name, strategy in self.strategies.items():
                    for ticker, data in market_data.items():
                        if data is None or len(data) < 50:
                            continue

                        if ticker in self.positions[name]:
                            self.update_position(
                                strategy, ticker, current_prices[ticker], data
                            )
                        else:
                            signal = strategy.check_entry(data, spy_data)
                            if signal:
                                self.enter_position(strategy, ticker, signal)

                self.log_status(current_prices)
                last_check = time.time()
        except KeyboardInterrupt:
            self.logger.info("\n🛑 Shutting down...")
            self.log_status({})

    def enter_position(self, strategy: Any, ticker: str, signal: Dict[str, Any]):
        """Enters a new long or short position."""
        name, price = strategy.name, signal["price"]
        pos_val = strategy.get_position_size(
            signal.get("conviction", 5),
            self.balances[name],
            signal.get("volatility", 0.02),
        )

        self.positions[name][ticker] = {
            "type": signal.get("type", "long"),
            "shares": (pos_val * strategy.leverage) / price,
            "entry_price": price,
            "position_value": pos_val,
            "entry_date": datetime.now(),
            "highest_price": price,
            "lowest_price": price,
            "bars_held": 0,
        }
        self.logger.info(
            f"📈 [{name.upper()}] {signal.get('type', 'long').upper()} {ticker} @ ${price:.2f}"
        )

    def exit_position(
        self, strategy: Any, ticker: str, current_price: float, reason: str
    ):
        """Exits a position and records the trade."""
        name, pos = strategy.name, self.positions[name][ticker]
        price_change = (
            (current_price / pos["entry_price"] - 1)
            if pos["type"] == "long"
            else (pos["entry_price"] / current_price - 1)
        )
        net_pnl = (
            price_change * pos["position_value"] * strategy.leverage
        ) * 0.998  # Approx 0.2% fees

        self.balances[name] += net_pnl
        self.trade_history[name].append({"pnl": net_pnl})

        self.logger.info(
            f"📉 [{name.upper()}] EXIT {ticker}: ${net_pnl:+,.2f} ({price_change * 100:+.2f}%) - {reason}"
        )
        del self.positions[name][ticker]

    def update_position(
        self, strategy: Any, ticker: str, price: float, data: pd.DataFrame
    ):
        """Updates a position's stats and checks for exit signals."""
        pos = self.positions[strategy.name][ticker]
        pos.update(
            {
                "highest_price": max(price, pos["highest_price"]),
                "lowest_price": min(price, pos.get("lowest_price", price)),
                "bars_held": pos["bars_held"] + 1,
            }
        )

        should_exit, reason = strategy.check_exit(pos, price)
        if (
            not should_exit
            and hasattr(strategy, "check_exit_signal")
            and strategy.check_exit_signal(data, pos["type"])
        ):
            should_exit, reason = True, "signal"

        if should_exit:
            self.exit_position(strategy, ticker, price, reason)

    def log_status(self, current_prices: Dict[str, float]):
        """Logs a consolidated status of all strategies."""
        strategy_values = self.balances.copy()
        position_details = {name: [] for name in self.strategies}

        for name, positions in self.positions.items():
            for ticker, pos in positions.items():
                if ticker in current_prices:
                    pnl_mult = 1 if pos["type"] == "long" else -1
                    pnl = (
                        (current_prices[ticker] - pos["entry_price"])
                        * pos["shares"]
                        * pnl_mult
                    )
                    strategy_values[name] += pos["position_value"] + pnl
                    unrealized_pct = (pnl / pos["position_value"]) * 100
                    position_details[name].append(
                        f"{ticker} {'S ' if pos['type'] == 'short' else ''}({unrealized_pct:+.1f}%)"
                    )

        total_value = sum(strategy_values.values())
        self.logger.info(
            f"💼 Total Value: ${total_value:,.0f} ({(total_value / self.initial_balance - 1) * 100:+.1f}%)"
        )

        for name in self.strategies.keys():
            trades = self.trade_history[name]
            pnl_str = ""
            if trades:
                wins = sum(1 for t in trades if t["pnl"] > 0)
                pnl_str = f"| Trades: {len(trades)} ({wins / len(trades) * 100:.0f}% W) | P&L: ${sum(t['pnl'] for t in trades):+,.0f}"

            pos_str = ", ".join(position_details[name]) or "None"
            self.logger.info(
                f"  [{name.upper()}] Val: ${strategy_values[name]:,.0f} | Pos: {pos_str} {pnl_str}"
            )
        self.logger.info("-" * 20)


if __name__ == "__main__":
    # Configuration
    TICKERS = ["NVDA", "AMD", "ETH-USD", "BTC-USD"]
    INITIAL_BALANCE = 100
    CHECK_INTERVAL = 3600

    strategies = [
        SwingStrategy(),
        RSIBidirectionalStrategy(),
        RSIThresholdStrategy(),
        RSICrossoverStrategy(),
        MomentumBreakoutStrategy(),
    ]

    print("\n" + "=" * 70 + "\n🎯 STARTING MULTI-STRATEGY TRADER\n" + "=" * 70)
    trader = PaperTrader(strategies, INITIAL_BALANCE)
    trader.run(TICKERS, CHECK_INTERVAL)

