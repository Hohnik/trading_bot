# Trading Bot

A Python framework for developing, backtesting, and paper trading quantitative trading strategies.

## Features

*   **Backtesting:** A powerful backtesting engine to evaluate strategy performance on historical data.
*   **Paper Trading:** A live paper trading simulation to test strategies in real-time without real money.
*   **Strategy Validation:** Tools for rigorous statistical validation of strategies, including Monte Carlo and walk-forward analysis.
*   **Extensible Strategy Library:** A modular framework to easily add new trading strategies.

## Available Strategies

The following strategies are currently implemented:

*   Swing Strategy
*   Mean Reversion Strategy
*   Momentum Strategy
*   RSI Bidirectional Strategy
*   RSI Crossover Strategy
*   RSI Threshold Strategy

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd trade_bot
    ```

2.  Install the dependencies from `pyproject.toml`:
    ```bash
    pip install .
    ```

## Usage

### Backtesting

To run a backtest of the swing strategy on multiple assets:

```bash
python backtest.py
```

This will generate performance metrics and charts in the `results` directory.

### Paper Trading

To run the paper trader with multiple strategies:

```bash
python paper_trader.py
```

The trader will log its activity to `paper_trader.log` and the console.

### Validation

To run statistical validation tests on the swing strategy:

```bash
python validate.py
```

This will perform Monte Carlo and walk-forward analysis and save the results in the `validation` directory.

## Dependencies

All project dependencies are listed in the `pyproject.toml` file.
