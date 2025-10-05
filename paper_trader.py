"""
Swing Strategy Paper Trader
Live paper trading for the multi-factor momentum swing strategy.

Monitors 4-hour data and executes trades based on strategy signals.
Logs all activity and tracks performance in real-time.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from swing_strategy import backtest_swing_strategy
import os


class SwingPaperTrader:
    """Paper trading system for swing strategy."""
    
    def __init__(self, initial_balance=10000, leverage=2.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.positions = {}  # {ticker: position_data}
        self.trade_history = []
        self.equity_curve = [initial_balance]
        
        # Strategy parameters
        self.base_position_size = 0.4  # 40% base position
        self.hold_min_bars = 4
        self.hold_max_bars = 30
        
        # Trading hours (4-hour bars update at specific times)
        self.update_times = ['09:30', '13:30', '17:30', '21:30']  # EST
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("swing_paper_trader.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🚀 Swing Paper Trader initialized with ${initial_balance:,.2f}")
    
    def get_live_data(self, ticker, interval='1h', period='5d'):
        """Get live market data."""
        try:
            data = yf.download(ticker, period=period, interval=interval,
                             auto_adjust=True, progress=False)
            
            if data.empty:
                return None
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            data.dropna(inplace=True)
            return data
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {e}")
            return None
    
    def convert_to_4h(self, hourly_data):
        """Convert 1-hour data to 4-hour bars."""
        if hourly_data is None or len(hourly_data) < 4:
            return None
        
        # Resample to 4-hour bars
        data_4h = hourly_data.resample('4H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        return data_4h
    
    def check_signals(self, ticker, data_4h, spy_data=None):
        """Check for entry/exit signals using swing strategy."""
        if data_4h is None or len(data_4h) < 100:
            return None, None
        
        # Run strategy on recent data
        try:
            equity_curve, trades = backtest_swing_strategy(
                data_4h.copy(), 
                spy_data,
                initial_balance=10000,
                leverage=self.leverage,
                base_position_size=self.base_position_size,
                hold_min_bars=self.hold_min_bars,
                hold_max_bars=self.hold_max_bars
            )
            
            if not trades:
                return None, None
            
            # Get the most recent trade
            latest_trade = trades[-1]
            
            # Check if this is a new trade (not already in our positions)
            if ticker in self.positions:
                # Check if it's the same trade (same entry date)
                if (self.positions[ticker]['entry_date'] == latest_trade['entry_date'] and
                    self.positions[ticker]['entry_price'] == latest_trade['entry_price']):
                    return None, None  # Same trade, no action needed
            
            # New trade signal
            if latest_trade['type'] == 'long':
                return 'BUY', latest_trade
            elif latest_trade['type'] == 'short':
                return 'SELL', latest_trade
                
        except Exception as e:
            self.logger.error(f"Error checking signals for {ticker}: {e}")
            return None, None
    
    def execute_trade(self, ticker, signal, trade_data, current_price):
        """Execute a paper trade."""
        if signal == 'BUY':
            # Calculate position size
            position_value = self.balance * self.base_position_size
            shares = (position_value * self.leverage) / current_price
            
            self.positions[ticker] = {
                'type': 'long',
                'shares': shares,
                'entry_price': current_price,
                'entry_date': datetime.now(),
                'position_value': position_value,
                'highest_price': current_price,
                'bars_held': 0,
                'entry_signals': trade_data.get('entry_signals', []),
                'conviction': trade_data.get('conviction', 0)
            }
            
            self.logger.info(f"📈 BUY {ticker}: {shares:.2f} shares at ${current_price:.2f} "
                           f"(Value: ${position_value:,.2f})")
            
        elif signal == 'SELL' and ticker in self.positions:
            # Close position
            position = self.positions[ticker]
            exit_value = position['shares'] * current_price
            pnl = (current_price - position['entry_price']) / position['entry_price'] * position['position_value'] * self.leverage
            
            # Calculate fees (0.1% per side)
            fees = position['position_value'] * self.leverage * 0.001 * 2
            net_pnl = pnl - fees
            
            self.balance += net_pnl
            
            # Record trade
            self.trade_history.append({
                'ticker': ticker,
                'type': position['type'],
                'entry_date': position['entry_date'],
                'exit_date': datetime.now(),
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'shares': position['shares'],
                'pnl': net_pnl,
                'return_pct': (current_price / position['entry_price'] - 1) * 100,
                'bars_held': position['bars_held'],
                'exit_reason': 'signal_exit'
            })
            
            self.logger.info(f"📉 SELL {ticker}: {position['shares']:.2f} shares at ${current_price:.2f} "
                           f"(PnL: ${net_pnl:,.2f}, Return: {((current_price/position['entry_price']-1)*100):.2f}%)")
            
            del self.positions[ticker]
    
    def update_positions(self, ticker, current_price):
        """Update position tracking (highest price, bars held)."""
        if ticker in self.positions:
            position = self.positions[ticker]
            
            # Update highest price for trailing stops
            if current_price > position['highest_price']:
                position['highest_price'] = current_price
            
            # Update bars held
            position['bars_held'] += 1
            
            # Check exit conditions
            should_exit = False
            exit_reason = None
            
            price_change = (current_price / position['entry_price'] - 1)
            drawdown_from_high = (current_price / position['highest_price'] - 1)
            
            # Take profit: +20%
            if price_change >= 0.20:
                should_exit = True
                exit_reason = 'profit_target'
            
            # Stop loss: -4%
            elif price_change <= -0.04:
                should_exit = True
                exit_reason = 'stop_loss'
            
            # Trailing stop: -6% from highest
            elif drawdown_from_high <= -0.06:
                should_exit = True
                exit_reason = 'trailing_stop'
            
            # Max hold time
            elif position['bars_held'] >= self.hold_max_bars:
                should_exit = True
                exit_reason = 'max_hold'
            
            # Min hold time check
            if should_exit and position['bars_held'] < self.hold_min_bars:
                if exit_reason != 'stop_loss':
                    should_exit = False
            
            if should_exit:
                self.execute_trade(ticker, 'SELL', None, current_price)
    
    def get_portfolio_value(self, current_prices):
        """Calculate total portfolio value."""
        total_value = self.balance
        
        for ticker, position in self.positions.items():
            if ticker in current_prices:
                current_price = current_prices[ticker]
                position_value = position['shares'] * current_price
                total_value += position_value
        
        return total_value
    
    def log_portfolio_status(self, current_prices):
        """Log current portfolio status."""
        portfolio_value = self.get_portfolio_value(current_prices)
        total_return = (portfolio_value / self.initial_balance - 1) * 100
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"💼 PORTFOLIO UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total Value: ${portfolio_value:,.2f}")
        self.logger.info(f"Cash Balance: ${self.balance:,.2f}")
        self.logger.info(f"Total Return: {total_return:+.2f}%")
        self.logger.info(f"Active Positions: {len(self.positions)}")
        
        if self.positions:
            for ticker, pos in self.positions.items():
                if ticker in current_prices:
                    current_price = current_prices[ticker]
                    unrealized_pnl = (current_price / pos['entry_price'] - 1) * pos['position_value'] * self.leverage
                    unrealized_pct = (current_price / pos['entry_price'] - 1) * 100
                    
                    self.logger.info(f"  {ticker}: {pos['shares']:.2f} shares @ ${current_price:.2f} "
                                   f"(Unrealized: ${unrealized_pnl:+.2f}, {unrealized_pct:+.2f}%)")
        
        if self.trade_history:
            recent_trades = self.trade_history[-5:]  # Last 5 trades
            self.logger.info(f"\nRecent Trades:")
            for trade in recent_trades:
                self.logger.info(f"  {trade['ticker']} {trade['type']}: "
                               f"{trade['return_pct']:+.2f}% (${trade['pnl']:+.2f})")
        
        self.logger.info(f"{'='*60}")
    
    def run(self, tickers=['NVDA', 'AMD', 'ETH-USD'], check_interval=3600):
        """
        Run the paper trader.
        
        Args:
            tickers: List of tickers to monitor
            check_interval: How often to check (seconds) - default 1 hour
        """
        self.logger.info(f"🚀 Starting Swing Paper Trader")
        self.logger.info(f"Monitoring: {', '.join(tickers)}")
        self.logger.info(f"Check interval: {check_interval/60:.1f} minutes")
        self.logger.info(f"Press Ctrl+C to stop")
        
        last_check_time = time.time()
        
        try:
            while True:
                current_time = time.time()
                
                # Check if it's time to update (every hour)
                if current_time - last_check_time >= check_interval:
                    self.logger.info(f"\n🔄 Checking markets at {datetime.now().strftime('%H:%M:%S')}")
                    
                    current_prices = {}
                    
                    for ticker in tickers:
                        # Get 1-hour data and convert to 4-hour
                        hourly_data = self.get_live_data(ticker, '1h', '5d')
                        data_4h = self.convert_to_4h(hourly_data)
                        
                        if data_4h is None:
                            self.logger.warning(f"No data for {ticker}")
                            continue
                        
                        current_price = data_4h['Close'].iloc[-1]
                        current_prices[ticker] = current_price
                        
                        # Get SPY data for relative strength
                        spy_hourly = self.get_live_data('SPY', '1h', '5d')
                        spy_4h = self.convert_to_4h(spy_hourly) if spy_hourly is not None else None
                        
                        # Update existing positions
                        self.update_positions(ticker, current_price)
                        
                        # Check for new signals
                        signal, trade_data = self.check_signals(ticker, data_4h, spy_4h)
                        
                        if signal:
                            self.execute_trade(ticker, signal, trade_data, current_price)
                    
                    # Log portfolio status
                    self.log_portfolio_status(current_prices)
                    
                    # Update equity curve
                    portfolio_value = self.get_portfolio_value(current_prices)
                    self.equity_curve.append(portfolio_value)
                    
                    last_check_time = current_time
                
                # Sleep for 1 minute before next check
                time.sleep(60)
                
        except KeyboardInterrupt:
            self.logger.info(f"\n🛑 Paper trader stopped by user")
            self.log_portfolio_status(current_prices)
            
            # Final summary
            if self.trade_history:
                total_pnl = sum([t['pnl'] for t in self.trade_history])
                winning_trades = [t for t in self.trade_history if t['pnl'] > 0]
                win_rate = len(winning_trades) / len(self.trade_history) * 100
                
                self.logger.info(f"\n📊 TRADING SUMMARY:")
                self.logger.info(f"Total Trades: {len(self.trade_history)}")
                self.logger.info(f"Win Rate: {win_rate:.1f}%")
                self.logger.info(f"Total P&L: ${total_pnl:,.2f}")
                self.logger.info(f"Final Return: {((portfolio_value/self.initial_balance-1)*100):+.2f}%")
        
        except Exception as e:
            self.logger.error(f"❌ Unexpected error: {e}", exc_info=True)


def main():
    """Main function to run the paper trader."""
    
    # Configuration
    TICKERS = ['NVDA', 'AMD', 'ETH-USD']  # Best performing assets
    INITIAL_BALANCE = 10000
    LEVERAGE = 2.0  # Conservative for paper trading
    CHECK_INTERVAL = 3600  # Check every hour
    
    print("\n" + "="*60)
    print("🚀 SWING STRATEGY PAPER TRADER")
    print("="*60)
    print(f"Tickers: {', '.join(TICKERS)}")
    print(f"Initial Balance: ${INITIAL_BALANCE:,}")
    print(f"Leverage: {LEVERAGE}x")
    print(f"Check Interval: {CHECK_INTERVAL/60:.1f} minutes")
    print("="*60)
    print("Press Ctrl+C to stop")
    print("="*60)
    
    # Create and run trader
    trader = SwingPaperTrader(initial_balance=INITIAL_BALANCE, leverage=LEVERAGE)
    trader.run(tickers=TICKERS, check_interval=CHECK_INTERVAL)


if __name__ == "__main__":
    main()