"""
STATISTICAL VALIDATION FOR TRADING STRATEGIES
Tests for statistical significance using rigorous methods.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import sys
from io import StringIO
from typing import Any, List

# Import from our backtester and strategies
from backtest import download_data_with_spy, backtest_strategy, calculate_metrics
from strategies.swing_strategy import SwingStrategy
from strategies.momentum_strategy import MomentumBreakoutStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.rsi_crossover_strategy import RSICrossoverStrategy
from strategies.rsi_threshold_strategy import RSIThresholdStrategy
from strategies.rsi_bidirectional_strategy import RSIBidirectionalStrategy
from strategies.macd_strategy import MACDStrategy
from strategies.bollinger_bands_strategy import BollingerBandsStrategy
from strategies.dual_thrust_strategy import DualThrustStrategy


def quiet_download_data(*args, **kwargs):
    """Wrapper to suppress stdout from the download function."""
    original_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        return download_data_with_spy(*args, **kwargs)
    finally:
        sys.stdout = original_stdout


def monte_carlo_test(strategy: Any, ticker: str, n_simulations: int = 500):
    """Monte Carlo permutation test to check if a strategy has a real edge."""
    print(f"  - Running Monte Carlo test...")
    data, spy_data = quiet_download_data(ticker, '4h', '180d')
    if data is None or len(data) < 50: return None

    equity_curve, trades = backtest_strategy(strategy, data.copy(), spy_data)
    metrics = calculate_metrics(equity_curve, trades, data)
    if not metrics: return None
    
    original_return = metrics['total_return']
    price_returns = data['Close'].pct_change().dropna()
    permuted_returns = []

    for _ in range(n_simulations):
        shuffled_returns = np.random.permutation(price_returns.values)
        new_prices = [data['Close'].iloc[0]]
        for r in shuffled_returns:
            new_prices.append(new_prices[-1] * (1 + r))

        shuffled_data = data.copy()
        shuffled_data['Close'] = new_prices
        
        try:
            eq, tr = backtest_strategy(strategy, shuffled_data, spy_data)
            m = calculate_metrics(eq, tr, shuffled_data)
            if m: permuted_returns.append(m['total_return'])
        except Exception:
            permuted_returns.append(0)
            
    p_value = np.sum(np.array(permuted_returns) >= original_return) / n_simulations
    status = "✅ SIGNIFICANT" if p_value < 0.05 else "⚠️ MARGINAL" if p_value < 0.10 else "❌ NOT SIG"

    os.makedirs("validation", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(permuted_returns, bins=50, alpha=0.7, color='blue', label='Simulated Returns')
    plt.axvline(original_return, color='red', ls='--', lw=2, label=f'Actual: {original_return:.2f}%')
    plt.title(f'{ticker} - {strategy.name.upper()} Monte Carlo (p={p_value:.4f})')
    plt.xlabel('Return (%)'); plt.ylabel('Frequency'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f'validation/{ticker}_{strategy.name}_monte_carlo.png', dpi=150)
    plt.close()
    
    return {'ticker': ticker, 'p_value': p_value, 'status': status}


def walk_forward_test(strategy: Any, ticker: str):
    """Walk-forward analysis to test robustness across different time periods."""
    print(f"  - Running Walk-Forward test...")
    data, spy_data = quiet_download_data(ticker, '4h', '365d')
    if data is None or len(data) < 200: return None
    
    n_periods = 4
    period_size = len(data) // n_periods
    results = []

    for i in range(n_periods):
        start, end = i * period_size, (i + 1) * period_size
        period_data = data.iloc[start:end]
        period_spy = spy_data.iloc[start:end] if spy_data is not None else None
        if len(period_data) < 50: continue
        
        eq, tr = backtest_strategy(strategy, period_data.copy(), period_spy)
        m = calculate_metrics(eq, tr, period_data)
        if m: results.append(m)

    if results:
        returns = [r['total_return'] for r in results]
        consistency = np.sum(np.array(returns) > 0) / len(returns) * 100
        status = "✅ CONSISTENT" if consistency >= 75 else "⚠️ INCONSISTENT"
        return {'ticker': ticker, 'consistency': f"{consistency:.0f}%", 'status': status}
    return None


def validate_strategy(strategy: Any, assets: List[str]):
    """Runs a full validation suite for a given strategy."""
    print(f"\n{'='*80}\n🔥 VALIDATING STRATEGY: {strategy.name.upper()}\n{'='*80}")
    
    all_results = []
    for ticker in assets:
        print(f"\nValidating on {ticker}...")
        mc = monte_carlo_test(strategy, ticker)
        wf = walk_forward_test(strategy, ticker)
        if mc and wf:
            all_results.append({
                'Asset': ticker,
                'MC p-value': mc['p_value'],
                'MC Status': mc['status'],
                'WF Consistency': wf['consistency'],
                'WF Status': wf['status']
            })

    if all_results:
        print(f"\n--- VALIDATION SUMMARY for {strategy.name.upper()} ---")
        df = pd.DataFrame(all_results).set_index('Asset')
        print(df)
        
        avg_p_value = df['MC p-value'].mean()
        print(f"\n  Average Monte Carlo p-value: {avg_p_value:.4f}")
        overall_status = "✅ STRONG EDGE" if avg_p_value < 0.05 else "⚠️ MODERATE EDGE" if avg_p_value < 0.10 else "❌ NO EDGE"
        print(f"  Overall Verdict: {overall_status}")

if __name__ == "__main__":
    assets_to_validate = ['NVDA', 'AMD', 'COIN', 'TSLA']
    
    strategies_to_validate = [
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

    for strategy in strategies_to_validate:
        validate_strategy(strategy, assets_to_validate)