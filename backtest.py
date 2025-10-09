"""
MULTI-STRATEGY BACKTESTER
A flexible backtesting engine for various trading strategies.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from typing import Any, List

# Import all strategies
from strategies.swing_strategy import SwingStrategy
from strategies.momentum_strategy import MomentumBreakoutStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.rsi_crossover_strategy import RSICrossoverStrategy
from strategies.rsi_threshold_strategy import RSIThresholdStrategy
from strategies.rsi_bidirectional_strategy import RSIBidirectionalStrategy
from strategies.macd_strategy import MACDStrategy
from strategies.bollinger_bands_strategy import BollingerBandsStrategy
from strategies.dual_thrust_strategy import DualThrustStrategy


def download_data_with_spy(ticker, interval='4h', period='180d'):
    """Download asset data along with SPY for relative strength."""
    print(f"Downloading {ticker} data ({interval})...")
    
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if data.empty: return None, None
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.droplevel(1)

    spy_data = yf.download('SPY', period=period, interval=interval, auto_adjust=True, progress=False)
    if not spy_data.empty and isinstance(spy_data.columns, pd.MultiIndex): spy_data.columns = spy_data.columns.droplevel(1)
    
    data.dropna(inplace=True)
    spy_data.dropna(inplace=True)
    
    return data, spy_data


def backtest_strategy(strategy: Any, data: pd.DataFrame, spy_data: pd.DataFrame, initial_balance: float = 10000):
    """A generic backtesting function for any strategy class."""
    balance = initial_balance
    equity_curve = [initial_balance]
    trades = []
    position = None

    for i in range(50, len(data)):
        current_data = data.iloc[:i]
        current_price = current_data['Close'].iloc[-1]

        if position:
            position['bars_held'] += 1
            position['highest_price'] = max(position.get('highest_price', current_price), current_price)
            position['lowest_price'] = min(position.get('lowest_price', current_price), current_price)

            should_exit, reason = strategy.check_exit(position, current_price)
            if not should_exit and hasattr(strategy, 'check_exit_signal'):
                if strategy.check_exit_signal(current_data, position['type']):
                    should_exit, reason = True, 'signal'

            if should_exit:
                price_change = (current_price / position['entry_price'] - 1) if position['type'] == 'long' else (position['entry_price'] / current_price - 1)
                pnl = (price_change * position['position_value'] * strategy.leverage) * 0.998
                balance += pnl
                
                trades.append({
                    'pnl': pnl, 'return_pct': price_change * 100, 'bars_held': position['bars_held'],
                    'entry_date': position['entry_date'], 'exit_date': current_data.index[-1],
                    'entry_price': position['entry_price'], 'exit_price': current_price,
                    'exit_reason': reason, 'entry_signals': position.get('entry_signals', [])
                })
                position = None
        
        if not position:
            signal = strategy.check_entry(current_data, spy_data)
            if signal:
                pos_val = strategy.get_position_size(signal.get('conviction', 5), balance)
                position = {
                    'type': signal['type'], 'shares': (pos_val * strategy.leverage) / current_price,
                    'entry_price': current_price, 'position_value': pos_val,
                    'entry_date': current_data.index[-1], 'highest_price': current_price,
                    'lowest_price': current_price, 'bars_held': 0,
                    'entry_signals': signal.get('entry_signals', [])
                }

        equity = balance
        if position:
            pnl_mult = 1 if position['type'] == 'long' else -1
            unrealized_pnl = (current_price - position['entry_price']) * position['shares'] * pnl_mult
            equity += unrealized_pnl
        equity_curve.append(equity)

    return equity_curve, trades


def calculate_metrics(equity_curve, trades, data):
    """Calculate comprehensive performance metrics."""
    if not equity_curve or len(equity_curve) < 2: return None
    
    initial, final = equity_curve[0], equity_curve[-1]
    total_return = (final / initial - 1) * 100
    
    eq_array = np.array(equity_curve)
    returns = np.diff(eq_array) / eq_array[:-1]
    
    periods_per_year = 252 * (6.5 / 4) # 4h bars
    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)) if np.std(returns) > 0 else 0
    
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.0001
    sortino = (np.mean(returns) / downside_std * np.sqrt(periods_per_year)) if downside_std > 0 else 0
    
    peak = np.maximum.accumulate(eq_array)
    max_dd = np.min((eq_array - peak) / peak) * 100 if len(peak) > 0 else 0
    
    if trades:
        wins = [t for t in trades if t['pnl'] > 0]
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        profit_factor = sum(t['pnl'] for t in wins) / abs(sum(t['pnl'] for t in trades if t['pnl'] < 0)) if any(t['pnl'] < 0 for t in trades) else 100
    else:
        win_rate, profit_factor = 0, 0

    bh_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
    
    return {
        'total_return': total_return, 'sharpe': sharpe, 'sortino': sortino, 'max_dd': max_dd,
        'win_rate': win_rate, 'profit_factor': profit_factor, 'num_trades': len(trades),
        'bh_return': bh_return
    }


def run_backtest(strategy: Any, ticker: str, interval='4h', period='180d', plot=True):
    """Run a backtest for a given strategy and ticker."""
    print(f"\n{'='*40}\nBACKTEST: {strategy.name.upper()} on {ticker}\n{'='*40}")
    
    data, spy_data = download_data_with_spy(ticker, interval, period)
    if data is None or len(data) < 50:
        print(f"Insufficient data for {ticker}")
        return None
    
    equity_curve, trades = backtest_strategy(strategy, data.copy(), spy_data)
    metrics = calculate_metrics(equity_curve, trades, data)
    if not metrics:
        print("No results")
        return None
    
    print_metrics(metrics)
    
    if plot:
        filename = f'results/{ticker}_{strategy.name}_{interval}.png'
        plot_results(ticker, strategy, equity_curve, trades, data, metrics, filename)
    
    return metrics

def print_metrics(metrics: dict):
    """Prints a formatted table of performance metrics."""
    print(f"  Total Return: {metrics['total_return']:>10.2f}%   |   Buy & Hold: {metrics['bh_return']:>10.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe']:>10.4f}   |   Sortino Ratio: {metrics['sortino']:>8.4f}")
    print(f"  Max Drawdown: {metrics['max_dd']:>10.2f}%   |   Win Rate: {metrics['win_rate']:>13.1f}%")
    print(f"  Profit Factor: {metrics['profit_factor']:>9.2f}   |   Total Trades: {metrics['num_trades']:>9}")

def plot_results(ticker, strategy, equity_curve, trades, data, metrics, filename):
    """Generates and saves a plot of the backtest results."""
    os.makedirs("results", exist_ok=True)
    fig = plt.figure(figsize=(16, 10))
    
    # Equity Curve
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(data.index[:len(equity_curve)], equity_curve, lw=2, color='darkgreen', label=f'Strategy: {metrics["total_return"]:.1f}%')
    bh_equity = [10000 * (c / data['Close'].iloc[0]) for c in data['Close'][:len(equity_curve)]]
    ax1.plot(data.index[:len(equity_curve)], bh_equity, lw=2, color='blue', alpha=0.6, ls='--', label=f'Buy & Hold: {metrics["bh_return"]:.1f}%')
    ax1.set_title(f'{ticker} - {strategy.name.upper()} | {metrics["num_trades"]} trades', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Price with Trades
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(data.index, data['Close'], color='gray', alpha=0.7, lw=1.5, label='Price')
    
    for t in trades:
        color = 'green' if t['pnl'] > 0 else 'red'
        size = min(120, max(40, abs(t['return_pct']) * 10))
        ax2.scatter(t['entry_date'], t['entry_price'], color=color, marker='^', s=size, alpha=0.9, edgecolors='black')
        ax2.scatter(t['exit_date'], t['exit_price'], color=color, marker='v', s=size, alpha=0.9, edgecolors='black')
    
    ax2.set_title('Price & Trades')
    ax2.set_ylabel('Price ($)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Chart saved: {filename}")

def test_strategy_on_multiple_assets(strategy: Any, assets: List[str]):
    """Tests a single strategy on a list of assets."""
    print(f"\n{'='*80}\n🔥 TESTING STRATEGY: {strategy.name.upper()}\n{'='*80}")
    results = []
    for ticker in assets:
        metrics = run_backtest(strategy, ticker, plot=True)
        if metrics:
            metrics['ticker'] = ticker
            results.append(metrics)
    
    if results:
        print(f"\n--- SUMMARY for {strategy.name.upper()} ---")
        df = pd.DataFrame(results)
        df = df.set_index('ticker')
        print(df[['total_return', 'sharpe', 'win_rate', 'num_trades']].round(2))
        print(f"\nAverage Return: {df['total_return'].mean():.2f}%")


if __name__ == "__main__":
    assets_to_test = ['NVDA', 'TSLA', 'AMD', 'COIN', 'BTC-USD', 'ETH-USD']
    
    strategies_to_test = [
        SwingStrategy(),
        MomentumBreakoutStrategy(),
        MeanReversionStrategy(),
        RSIBidirectionalStrategy(),
        RSICrossoverStrategy(),
        RSIThresholdStrategy(),
        MACDStrategy(),
        BollingerBandsStrategy(),
        DualThrustStrategy(),
    ]

    for strategy in strategies_to_test:
        test_strategy_on_multiple_assets(strategy, assets_to_test)
