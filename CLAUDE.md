# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python framework for developing, backtesting, and paper trading quantitative trading strategies on crypto and equity assets using 4-hour timeframes.

## Commands

### Development
```bash
# Install dependencies
uv sync

# Run backtest on all strategies
python backtest.py

# Run statistical validation (Monte Carlo & walk-forward analysis)
python validate.py

# Start paper trading (logs to paper_trader.log and console)
python paper_trader.py
```

## Architecture

### Data Flow
1. **Data Acquisition**: `yfinance` downloads historical data for assets and SPY (for relative strength calculations)
2. **Resampling**: Data is resampled to 4-hour intervals (see `paper_trader.py:57-68`)
3. **Strategy Execution**: Each strategy analyzes data and generates entry/exit signals
4. **Position Management**: Backtest engine or paper trader manages positions based on signals

### Strategy System

**Strategy Interface** (in `strategies/` directory):
- Required methods:
  - `check_entry(data, spy_data)` → Returns signal dict with `{'type': 'long'|'short', 'price': float, 'conviction': int}` or `None`
  - `check_exit(position, current_price)` → Returns `(bool, reason_string)`
  - `get_position_size(conviction, balance, volatility)` → Returns position size in dollars
- Properties:
  - `name`: string identifier for the strategy
  - `leverage`: leverage multiplier (typically 1.0-2.0)
- Shared indicators are in `strategies/indicators.py`

**When creating new strategies**, use the pattern from `strategies/swing_strategy.py` as a template.

### Backtesting Engine

Located in `backtest.py`:
- Generic backtesting function `backtest_strategy()` works with any strategy implementing the strategy interface
- Handles both long and short positions
- Applies 0.2% total trading fees (0.998 multiplier on P&L)
- Tracks comprehensive position metrics (bars held, highest/lowest prices, entry signals)
- Exit logic checks both strategy exit rules and optional signal-based exits

### Validation Framework

Located in `validate.py`:
- **Monte Carlo Test**: Permutation test (500 simulations) to check statistical significance (p-value < 0.05 = significant edge)
- **Walk-Forward Analysis**: Tests strategy across 4 time periods to verify consistency (≥75% profitable periods = consistent)
- Generates validation charts in `validation/` directory

### Paper Trading

Located in `paper_trader.py`:
- Multi-strategy paper trader that allocates separate capital to each strategy
- Fetches 1-hour data and resamples to 4-hour bars for strategy compatibility
- Runs on hourly check intervals (configurable via `check_interval` parameter)
- Tracks per-strategy balances, positions, and trade history
- Logs to both file (`paper_trader.log`) and console

### Key Configuration Parameters

**Strategy Parameters** (configurable in strategy `__init__`):
- `leverage`: Position leverage multiplier (default 2.0 for swing strategy)
- `position_size`: Base position size as fraction of balance (0.1-0.7)
- `profit_target`: Take profit threshold (e.g., 0.20 = 20%)
- `stop_loss`: Maximum loss threshold (e.g., 0.04 = 4%)
- `trailing_stop`: Trailing stop from peak (e.g., 0.06 = 6%)
- `hold_max_bars`: Maximum position duration in 4-hour bars

**Backtesting Parameters**:
- Default timeframe: 4-hour bars
- Default period: 180 days for backtesting, 365 days for walk-forward validation
- Initial balance: $10,000 for backtesting
- Warm-up period: 50 bars minimum before strategy starts trading

### Performance Metrics

All metrics calculated in `backtest.calculate_metrics()`:
- Total Return, Sharpe Ratio, Sortino Ratio
- Maximum Drawdown, Win Rate, Profit Factor
- Number of trades, Buy & Hold comparison
- Periods per year calculated as: `252 * (6.5 / 4)` for 4-hour bars

### Output Directories

- `results/`: Backtest charts (equity curves, trade markers)
- `validation/`: Monte Carlo distribution charts
- `paper_trader.log`: Paper trading activity log

## Available Strategies

**Active Strategies** (in `strategies/`):
- SwingStrategy: Multi-factor momentum with EMA, RSI, volume, and relative strength
- MomentumBreakoutStrategy
- MeanReversionStrategy
- RSIBidirectionalStrategy: Can go long or short based on RSI
- RSICrossoverStrategy
- RSIThresholdStrategy
