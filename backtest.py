"""
SWING STRATEGY BACKTESTER
Tests innovative multi-factor swing trading system
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from swing_strategy import backtest_swing_strategy
from datetime import datetime


def download_data_with_spy(ticker, interval='1h', period='180d'):
    """Download asset data along with SPY for relative strength."""
    print(f"Downloading {ticker} data ({interval})...")
    
    # Download main asset
    data = yf.download(ticker, period=period, interval=interval,
                       auto_adjust=True, progress=False)
    
    if data.empty:
        return None, None
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    # Download SPY for relative strength
    spy_data = yf.download('SPY', period=period, interval=interval,
                          auto_adjust=True, progress=False)
    
    if not spy_data.empty and isinstance(spy_data.columns, pd.MultiIndex):
        spy_data.columns = spy_data.columns.droplevel(1)
    
    data.dropna(inplace=True)
    spy_data.dropna(inplace=True)
    
    return data, spy_data


def calculate_metrics(equity_curve, trades, data):
    """Calculate comprehensive performance metrics."""
    if not equity_curve or len(equity_curve) < 2:
        return None
    
    initial = equity_curve[0]
    final = equity_curve[-1]
    total_return = (final / initial - 1) * 100
    
    # Returns analysis
    eq_array = np.array(equity_curve)
    returns = np.diff(eq_array) / eq_array[:-1]
    
    # Annualized Sharpe (assuming 4h bars = 1560 periods per year)
    periods_per_year = 1560  # 252 days * 6.5 hours / 4h bars
    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year) 
             if np.std(returns) > 0 else 0)
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.0001
    sortino = (np.mean(returns) / downside_std * np.sqrt(periods_per_year) 
              if downside_std > 0 else 0)
    
    # Drawdown
    peak = np.maximum.accumulate(eq_array)
    dd = (eq_array - peak) / peak
    max_dd = np.min(dd) * 100
    
    # Trade statistics
    if trades:
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] < 0]
        
        win_rate = len(wins) / len(trades) * 100
        avg_win = np.mean([t['return_pct'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['return_pct'] for t in losses]) if losses else 0
        
        total_win_pnl = sum([t['pnl'] for t in wins])
        total_loss_pnl = sum([abs(t['pnl']) for t in losses])
        profit_factor = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else 0
        
        avg_bars = np.mean([t['bars_held'] for t in trades])
        avg_hold_days = avg_bars / 6  # 4h bars to days
        
        # Consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for t in trades:
            if t['pnl'] > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        # Time analysis
        time_span = (data.index[-1] - data.index[0]).total_seconds() / (24 * 3600)
        trades_per_week = (len(trades) / time_span) * 7 if time_span > 0 else 0
        
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0
        avg_hold_days = 0
        max_consecutive_wins = max_consecutive_losses = 0
        trades_per_week = 0
    
    # Buy and hold
    bh_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'num_trades': len(trades),
        'avg_hold_days': avg_hold_days,
        'trades_per_week': trades_per_week,
        'bh_return': bh_return,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses
    }


def run_backtest(ticker, interval='4h', period='180d', plot=True):
    """
    Run swing strategy backtest.
    
    Using 4-hour data for swing trading (holds 2-7 days).
    """
    print(f"\n{'='*80}")
    print(f"SWING TRADING BACKTEST: {ticker}")
    print(f"{'='*80}")
    
    # Download data
    data, spy_data = download_data_with_spy(ticker, interval, period)
    
    if data is None or len(data) < 200:
        print(f"Insufficient data for {ticker}")
        return None
    
    print(f"Loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")
    
    # Run backtest
    equity_curve, trades = backtest_swing_strategy(data.copy(), spy_data)
    
    # Calculate metrics
    metrics = calculate_metrics(equity_curve, trades, data)
    
    if not metrics:
        print("No results")
        return None
    
    # Print results
    print(f"\n{'='*80}")
    print("PERFORMANCE METRICS")
    print(f"{'='*80}")
    print(f"  Total Return:          {metrics['total_return']:>10.2f}%")
    print(f"  Buy & Hold:            {metrics['bh_return']:>10.2f}%")
    print(f"  Alpha (vs B&H):        {metrics['total_return']-metrics['bh_return']:>10.2f}%")
    print(f"  Sharpe Ratio:          {metrics['sharpe']:>10.4f}")
    print(f"  Sortino Ratio:         {metrics['sortino']:>10.4f}")
    print(f"  Max Drawdown:          {metrics['max_dd']:>10.2f}%")
    print(f"\n  Total Trades:          {metrics['num_trades']:>10}")
    print(f"  Trades per Week:       {metrics['trades_per_week']:>10.2f}")
    print(f"  Win Rate:              {metrics['win_rate']:>10.1f}%")
    print(f"  Profit Factor:         {metrics['profit_factor']:>10.2f}")
    print(f"  Avg Win:               {metrics['avg_win']:>10.2f}%")
    print(f"  Avg Loss:              {metrics['avg_loss']:>10.2f}%")
    print(f"  Risk/Reward:           {abs(metrics['avg_win']/metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0:>10.2f}")
    print(f"  Avg Hold:              {metrics['avg_hold_days']:>10.1f} days")
    print(f"  Max Consecutive Wins:  {metrics['max_consecutive_wins']:>10}")
    print(f"  Max Consecutive Loss:  {metrics['max_consecutive_losses']:>10}")
    
    # Entry signal analysis
    if trades:
        all_signals = {}
        for t in trades:
            for signal in t['entry_signals']:
                all_signals[signal] = all_signals.get(signal, 0) + 1
        
        print(f"\n  Entry Signal Frequency:")
        for signal, count in sorted(all_signals.items(), key=lambda x: x[1], reverse=True):
            print(f"    {signal:30s} {count:>3} ({count/len(trades)*100:>5.1f}%)")
        
        # Exit reason analysis
        exit_reasons = {}
        for t in trades:
            reason = t['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        print(f"\n  Exit Reasons:")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"    {reason:30s} {count:>3} ({count/len(trades)*100:>5.1f}%)")
    
    # Plot
    if plot:
        os.makedirs("results", exist_ok=True)
        
        fig = plt.figure(figsize=(16, 12))
        
        # Equity curve
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(data.index[:len(equity_curve)], equity_curve,
                linewidth=2, color='darkgreen', label=f'Strategy: {metrics["total_return"]:.1f}%')
        
        # Buy & hold comparison
        bh_equity = [10000 * (data['Close'].iloc[i] / data['Close'].iloc[0]) 
                     for i in range(len(equity_curve))]
        ax1.plot(data.index[:len(equity_curve)], bh_equity,
                linewidth=2, color='blue', alpha=0.6, linestyle='--',
                label=f'Buy & Hold: {metrics["bh_return"]:.1f}%')
        
        ax1.set_title(f'{ticker} - Swing Trading Strategy | {metrics["num_trades"]} trades, {metrics["trades_per_week"]:.1f}/week',
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Price with trades
        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(data.index, data['Close'], color='gray', alpha=0.5, linewidth=1.5)
        
        for t in trades:
            color = 'green' if t['pnl'] > 0 else 'red'
            size = min(120, max(40, abs(t['return_pct']) * 5))
            
            ax2.scatter(t['entry_date'], t['entry_price'],
                       color=color, marker='^', s=size, alpha=0.7,
                       edgecolors='black', linewidth=0.5)
            ax2.scatter(t['exit_date'], t['exit_price'],
                       color=color, marker='v', s=size, alpha=0.7,
                       edgecolors='black', linewidth=0.5)
        
        ax2.set_title('Price & Trade Entries (^) / Exits (v)', fontsize=12)
        ax2.set_ylabel('Price ($)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Drawdown chart
        ax3 = plt.subplot(3, 1, 3)
        eq_array = np.array(equity_curve)
        peak = np.maximum.accumulate(eq_array)
        dd = (eq_array - peak) / peak * 100
        ax3.fill_between(data.index[:len(equity_curve)], 0, dd,
                        color='red', alpha=0.3)
        ax3.plot(data.index[:len(equity_curve)], dd,
                color='darkred', linewidth=1.5)
        ax3.set_title(f'Drawdown (Max: {metrics["max_dd"]:.2f}%)', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_ylabel('Drawdown (%)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'results/{ticker}_swing_{interval}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n📊 Chart saved: {filename}")
    
    return metrics


def test_multiple_assets():
    """Test on multiple assets."""
    
    # Focus on volatile tech stocks and crypto for swing trading
    assets = [
        'NVDA',    # High volatility tech
        'TSLA',    # High volatility EV
        'AMD',     # Tech
        'COIN',    # Crypto stock
        'BTC-USD', # Bitcoin
        'ETH-USD', # Ethereum
    ]
    
    results = []
    
    print("\n" + "="*80)
    print("TESTING SWING STRATEGY ON MULTIPLE ASSETS")
    print("="*80)
    
    for ticker in assets:
        metrics = run_backtest(ticker, interval='4h', period='180d', plot=True)
        if metrics:
            metrics['ticker'] = ticker
            results.append(metrics)
    
    # Summary
    if results:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"\n{'Ticker':<10} {'Return':<10} {'vs B&H':<10} {'Sharpe':<10} {'Win%':<8} "
              f"{'Trades':<8} {'T/Week':<8} {'PF':<8}")
        print("-"*80)
        
        for r in results:
            alpha = r['total_return'] - r['bh_return']
            print(f"{r['ticker']:<10} {r['total_return']:>8.2f}% {alpha:>8.2f}% "
                  f"{r['sharpe']:>8.4f} {r['win_rate']:>6.1f}% {r['num_trades']:>6} "
                  f"{r['trades_per_week']:>6.2f} {r['profit_factor']:>6.2f}")
        
        # Averages
        print("-"*80)
        avg_return = np.mean([r['total_return'] for r in results])
        avg_sharpe = np.mean([r['sharpe'] for r in results])
        avg_win_rate = np.mean([r['win_rate'] for r in results])
        avg_pf = np.mean([r['profit_factor'] for r in results])
        
        print(f"{'AVERAGE':<10} {avg_return:>8.2f}% {'':>10} {avg_sharpe:>8.4f} "
              f"{avg_win_rate:>6.1f}% {'':>14} {avg_pf:>6.2f}")
    
    return results


if __name__ == "__main__":
    print("\n🚀 INNOVATIVE SWING TRADING STRATEGY")
    print("="*80)
    print("Multi-factor confluence system with:")
    print("  • Multi-timeframe momentum")
    print("  • Volatility breakout detection")
    print("  • Volume confirmation")
    print("  • Relative strength vs SPY")
    print("  • Dynamic position sizing")
    print("  • Adaptive trailing stops")
    print("="*80)
    
    # Test on multiple assets
    test_multiple_assets()