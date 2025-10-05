"""
STATISTICAL VALIDATION FOR SWING STRATEGY
Tests for statistical significance using rigorous methods
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
from swing_strategy import backtest_swing_strategy
from backtest import download_data_with_spy, calculate_metrics


def monte_carlo_test(ticker, n_simulations=1000):
    """
    Monte Carlo permutation test.
    Randomly shuffles returns to test if strategy has real edge.
    """
    print(f"\n{'='*80}")
    print(f"MONTE CARLO TEST: {ticker}")
    print(f"{'='*80}")
    
    # Download data
    data, spy_data = download_data_with_spy(ticker, '4h', '180d')
    if data is None or len(data) < 200:
        return None
    
    # Run actual strategy
    equity_curve, trades = backtest_swing_strategy(data.copy(), spy_data)
    metrics = calculate_metrics(equity_curve, trades, data)
    
    if not metrics:
        return None
    
    original_return = metrics['total_return']
    original_sharpe = metrics['sharpe']
    
    print(f"Original Strategy:")
    print(f"  Return: {original_return:.2f}%")
    print(f"  Sharpe: {original_sharpe:.4f}")
    
    # Run permutations
    print(f"\nRunning {n_simulations} Monte Carlo simulations...")
    
    price_returns = data['Close'].pct_change().dropna()
    permuted_returns = []
    permuted_sharpes = []
    
    for i in range(n_simulations):
        if i % 100 == 0:
            print(f"  Simulation {i+1}/{n_simulations}...")
        
        # Shuffle returns
        shuffled_returns = np.random.permutation(price_returns.values)
        
        # Reconstruct price series
        shuffled_prices = [data['Close'].iloc[0]]
        for ret in shuffled_returns:
            shuffled_prices.append(shuffled_prices[-1] * (1 + ret))
        
        shuffled_data = data.copy()
        shuffled_data['Close'] = shuffled_prices[:len(shuffled_data)]
        shuffled_data['High'] = shuffled_data['Close'] * 1.01
        shuffled_data['Low'] = shuffled_data['Close'] * 0.99
        
        try:
            eq, tr = backtest_swing_strategy(shuffled_data, spy_data)
            m = calculate_metrics(eq, tr, shuffled_data)
            if m:
                permuted_returns.append(m['total_return'])
                permuted_sharpes.append(m['sharpe'])
        except:
            permuted_returns.append(0)
            permuted_sharpes.append(0)
    
    # Calculate p-values
    p_value_return = np.sum(np.array(permuted_returns) >= original_return) / n_simulations
    p_value_sharpe = np.sum(np.array(permuted_sharpes) >= original_sharpe) / n_simulations
    
    percentile_return = stats.percentileofscore(permuted_returns, original_return)
    percentile_sharpe = stats.percentileofscore(permuted_sharpes, original_sharpe)
    
    print(f"\nResults:")
    print(f"  Mean Permuted Return: {np.mean(permuted_returns):.2f}%")
    print(f"  P-value (Return): {p_value_return:.4f}")
    print(f"  Percentile (Return): {percentile_return:.1f}%")
    print(f"\n  Mean Permuted Sharpe: {np.mean(permuted_sharpes):.4f}")
    print(f"  P-value (Sharpe): {p_value_sharpe:.4f}")
    print(f"  Percentile (Sharpe): {percentile_sharpe:.1f}%")
    
    if p_value_return < 0.05:
        print(f"\n  ✅ STATISTICALLY SIGNIFICANT (p < 0.05)")
    elif p_value_return < 0.10:
        print(f"\n  ⚠️  MARGINALLY SIGNIFICANT (p < 0.10)")
    else:
        print(f"\n  ❌ NOT SIGNIFICANT (p >= 0.10)")
    
    # Plot
    os.makedirs("validation", exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(permuted_returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(original_return, color='red', linestyle='--', linewidth=2, label='Actual Strategy')
    ax1.set_xlabel('Return (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'{ticker} - Returns Distribution\np={p_value_return:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(permuted_sharpes, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(original_sharpe, color='red', linestyle='--', linewidth=2, label='Actual Strategy')
    ax2.set_xlabel('Sharpe Ratio')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{ticker} - Sharpe Distribution\np={p_value_sharpe:.4f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'validation/{ticker}_monte_carlo.png', dpi=300)
    plt.close()
    
    return {
        'ticker': ticker,
        'p_value_return': p_value_return,
        'p_value_sharpe': p_value_sharpe,
        'percentile_return': percentile_return,
        'percentile_sharpe': percentile_sharpe,
        'original_return': original_return,
        'original_sharpe': original_sharpe
    }


def walk_forward_test(ticker):
    """
    Walk-forward analysis.
    Tests robustness across different time periods.
    """
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD TEST: {ticker}")
    print(f"{'='*80}")
    
    # Download longer period
    data, spy_data = download_data_with_spy(ticker, '4h', '365d')
    if data is None or len(data) < 300:
        print("Insufficient data")
        return None
    
    # Split into periods
    n_periods = 4
    period_size = len(data) // n_periods
    
    period_results = []
    
    for i in range(n_periods):
        start_idx = i * period_size
        end_idx = min((i + 1) * period_size, len(data))
        
        period_data = data.iloc[start_idx:end_idx].copy()
        period_spy = spy_data.iloc[start_idx:end_idx].copy() if spy_data is not None else None
        
        if len(period_data) < 100:
            continue
        
        print(f"\nPeriod {i+1}: {period_data.index[0].date()} to {period_data.index[-1].date()}")
        
        try:
            eq, tr = backtest_swing_strategy(period_data, period_spy)
            m = calculate_metrics(eq, tr, period_data)
            
            if m:
                period_results.append({
                    'period': i + 1,
                    'return': m['total_return'],
                    'sharpe': m['sharpe'],
                    'win_rate': m['win_rate'],
                    'trades': m['num_trades']
                })
                
                print(f"  Return: {m['total_return']:.2f}%")
                print(f"  Sharpe: {m['sharpe']:.4f}")
                print(f"  Win Rate: {m['win_rate']:.1f}%")
                print(f"  Trades: {m['num_trades']}")
        except Exception as e:
            print(f"  Error: {e}")
    
    if period_results:
        returns = [p['return'] for p in period_results]
        consistency = np.sum(np.array(returns) > 0) / len(returns) * 100
        
        print(f"\nSummary:")
        print(f"  Consistency (% positive): {consistency:.1f}%")
        print(f"  Mean Return: {np.mean(returns):.2f}%")
        print(f"  Std Dev: {np.std(returns):.2f}%")
        
        if consistency >= 75:
            print(f"\n  ✅ HIGHLY CONSISTENT (>= 75%)")
        elif consistency >= 60:
            print(f"\n  ✅ CONSISTENT (>= 60%)")
        else:
            print(f"\n  ⚠️  INCONSISTENT (< 60%)")
    
    return {
        'ticker': ticker,
        'consistency': consistency if period_results else 0,
        'periods': period_results
    }


def validate_multiple_assets():
    """Run validation on multiple assets."""
    
    assets = ['NVDA', 'AMD', 'COIN']
    
    print("\n" + "="*80)
    print("STATISTICAL VALIDATION SUITE")
    print("="*80)
    print("Running Monte Carlo and Walk-Forward tests...")
    print("="*80)
    
    mc_results = []
    wf_results = []
    
    for ticker in assets:
        # Monte Carlo test
        mc = monte_carlo_test(ticker, n_simulations=500)  # 500 for speed
        if mc:
            mc_results.append(mc)
        
        # Walk-forward test
        wf = walk_forward_test(ticker)
        if wf:
            wf_results.append(wf)
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    print(f"\n{'Asset':<10} {'Return':<10} {'Sharpe':<10} {'P-val(R)':<10} {'P-val(S)':<10} {'Status':<15}")
    print("-"*80)
    
    for mc in mc_results:
        status = "✅ SIGNIFICANT" if mc['p_value_return'] < 0.05 else "⚠️  MARGINAL" if mc['p_value_return'] < 0.10 else "❌ NOT SIG"
        print(f"{mc['ticker']:<10} {mc['original_return']:>8.2f}% {mc['original_sharpe']:>8.4f} "
              f"{mc['p_value_return']:>8.4f} {mc['p_value_sharpe']:>8.4f} {status:<15}")
    
    print("\n" + "-"*80)
    
    sig_count = sum(1 for mc in mc_results if mc['p_value_return'] < 0.05)
    marg_count = sum(1 for mc in mc_results if 0.05 <= mc['p_value_return'] < 0.10)
    
    print(f"\nStatistically Significant: {sig_count}/{len(mc_results)}")
    print(f"Marginally Significant: {marg_count}/{len(mc_results)}")
    
    avg_p_return = np.mean([mc['p_value_return'] for mc in mc_results])
    avg_p_sharpe = np.mean([mc['p_value_sharpe'] for mc in mc_results])
    
    print(f"\nAverage P-values:")
    print(f"  Returns: {avg_p_return:.4f}")
    print(f"  Sharpe: {avg_p_sharpe:.4f}")
    
    if avg_p_return < 0.05:
        print(f"\n🎯 OVERALL VERDICT: Strategy shows STRONG statistical significance")
    elif avg_p_return < 0.10:
        print(f"\n⚠️  OVERALL VERDICT: Strategy shows MODERATE statistical significance")
    else:
        print(f"\n❌ OVERALL VERDICT: Strategy lacks statistical significance")


if __name__ == "__main__":
    validate_multiple_assets()