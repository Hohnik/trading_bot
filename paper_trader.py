import yfinance as yf
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("paper_trader.log"),
                        logging.StreamHandler()
                    ])

# --- Indicator Functions ---
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
    trend = (prices.iloc[-1] - prices.iloc[-1 - period]) / prices.iloc[-1 - period] * 100
    return abs(trend)

# --- Simulated Exchange ---
class SimulatedExchange:
    def __init__(self, initial_balance=10000, fee_percent=0.001):
        self.balance_usd = initial_balance
        self.portfolio = {}
        self.fee_percent = fee_percent
        self.trades = []
        logging.info(f"Simulated exchange initialized with ${initial_balance:,.2f} USD.")

    def get_portfolio_value(self, current_prices):
        total_value = self.balance_usd
        for symbol, amount in self.portfolio.items():
            total_value += amount * current_prices.get(symbol, 0)
        return total_value

    def execute_buy(self, symbol, amount_usd, current_price):
        if self.balance_usd < amount_usd:
            logging.warning("Insufficient funds to execute buy.")
            return False
        
        fee = amount_usd * self.fee_percent
        amount_to_buy = (amount_usd - fee) / current_price
        
        self.balance_usd -= amount_usd
        self.portfolio[symbol] = self.portfolio.get(symbol, 0) + amount_to_buy
        
        trade_log = f"BUY: {amount_to_buy:.6f} {symbol} at ${current_price:,.2f} (Cost: ${amount_usd:,.2f}, Fee: ${fee:,.2f})"
        self.trades.append(trade_log)
        logging.info(trade_log)
        return True

    def execute_sell(self, symbol, amount_asset, current_price):
        if self.portfolio.get(symbol, 0) < amount_asset:
            logging.warning("Insufficient assets to execute sell.")
            return False

        self.portfolio[symbol] -= amount_asset
        amount_usd = amount_asset * current_price
        fee = amount_usd * self.fee_percent
        self.balance_usd += amount_usd - fee

        trade_log = f"SELL: {amount_asset:.6f} {symbol} at ${current_price:,.2f} (Value: ${amount_usd:,.2f}, Fee: ${fee:,.2f})"
        self.trades.append(trade_log)
        logging.info(trade_log)
        return True

# --- Live Data Fetching ---
def get_live_data(ticker, period="3d", interval="15m"):
    try:
        return yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# --- Helper Functions ---
def log_portfolio_update(exchange, current_prices):
    portfolio_value = exchange.get_portfolio_value(current_prices)
    holdings_str = ", ".join([f"{amount:.6f} {symbol}" for symbol, amount in exchange.portfolio.items() if amount > 0])
    if not holdings_str:
        holdings_str = "None"
    
    log_msg = (
        f"PORTFOLIO UPDATE | Total Value: ${portfolio_value:,.2f} | "
        f"USD Balance: ${exchange.balance_usd:,.2f} | Holdings: {holdings_str}"
    )
    logging.info(log_msg)
    logging.info("-" * 80)

# --- Main Trading Logic ---
def main():
    best_params = {'rsi_period': 10, 'rsi_confirmation_threshold': 4, 'min_trend_strength': 0.2}
    logging.info(f"Using optimized parameters: {best_params}")

    tickers = ['LTC-USD', 'ETH-USD', 'BTC-USD']
    exchange = SimulatedExchange(initial_balance=10000)
    
    positions = {ticker: None for ticker in tickers}
    last_heartbeat_time = time.time()

    logging.info("--- Starting Paper Trading Bot ---")
    try:
        while True:
            current_prices = {}
            trade_occurred = False

            for ticker in tickers:
                data = get_live_data(ticker)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)

                if data.empty:
                    logging.warning(f"Could not fetch data for {ticker}. Skipping.")
                    continue
                
                current_price = data['Close'].iloc[-1]
                current_prices[ticker] = current_price

                # --- Signal Generation ---
                data['rsi'] = calculate_rsi_manual(data['Close'], period=best_params['rsi_period'])
                
                current_rsi = data['rsi'].iloc[-1]
                previous_rsi = data['rsi'].iloc[-2]

                if pd.isna(current_rsi) or pd.isna(previous_rsi):
                    continue

                # --- Entry Logic ---
                if positions[ticker] is None:
                    rsi_crossed_above = previous_rsi <= 50 and current_rsi > 50
                    rsi_confirmation = current_rsi >= (50 + best_params['rsi_confirmation_threshold'])
                    
                    if rsi_crossed_above and rsi_confirmation:
                        logging.info(f"Potential BUY signal for {ticker} at ${current_price:,.2f}")
                        position_size_usd = exchange.balance_usd * 0.20
                        if exchange.execute_buy(ticker, position_size_usd, current_price):
                            positions[ticker] = {'type': 'long', 'entry_price': current_price}
                            trade_occurred = True

                # --- Exit Logic ---
                elif positions[ticker]['type'] == 'long':
                    rsi_crossed_below = previous_rsi >= 50 and current_rsi < 50
                    rsi_confirmation = current_rsi <= (50 - best_params['rsi_confirmation_threshold'])

                    if rsi_crossed_below and rsi_confirmation:
                        logging.info(f"Potential SELL signal for {ticker} at ${current_price:,.2f}")
                        amount_to_sell = exchange.portfolio.get(ticker, 0)
                        if amount_to_sell > 0:
                            if exchange.execute_sell(ticker, amount_to_sell, current_price):
                                positions[ticker] = None
                                trade_occurred = True

            # --- Logging ---
            if trade_occurred:
                log_portfolio_update(exchange, current_prices)
            
            if time.time() - last_heartbeat_time > 3600:
                logging.info("--- Hourly Heartbeat ---")
                log_portfolio_update(exchange, current_prices)
                last_heartbeat_time = time.time()

            time.sleep(60)

    except KeyboardInterrupt:
        logging.info("--- Paper trading bot stopped by user. ---")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()