# Swing Strategy Paper Trader - Usage Guide

## 🚀 Quick Start

### Run the Paper Trader:
```bash
python paper_trader.py
```

**What it does:**
- Monitors NVDA, AMD, ETH-USD every hour
- Uses 4-hour swing strategy signals
- Executes paper trades automatically
- Logs everything to `swing_paper_trader.log`

### Test First (Optional):
```bash
python quick_test.py
```

**What it does:**
- Tests data fetching
- Verifies 4-hour conversion
- Checks signal detection
- Confirms everything works

## 📊 How It Works

### Monitoring Process:
1. **Every hour** - Checks for new 4-hour bars
2. **Converts** 1-hour data to 4-hour bars
3. **Runs strategy** on recent data
4. **Detects signals** - BUY/SELL based on strategy
5. **Executes trades** - Paper trades only (no real money)
6. **Updates positions** - Tracks stops, targets, trailing stops

### Strategy Parameters:
- **Position Size:** 40% of balance per trade
- **Leverage:** 2x (conservative for paper trading)
- **Stop Loss:** -4%
- **Take Profit:** +20%
- **Trailing Stop:** -6% from highest
- **Hold Time:** 1-7.5 days (4-30 bars)

## 📈 What You'll See

### Console Output:
```
🚀 SWING STRATEGY PAPER TRADER
============================================================
Tickers: NVDA, AMD, ETH-USD
Initial Balance: $10,000
Leverage: 2x
Check Interval: 60.0 minutes
============================================================

🔄 Checking markets at 14:30:00
📈 BUY NVDA: 21.33 shares at $187.62 (Value: $4,000.00)

💼 PORTFOLIO UPDATE - 2025-10-05 14:30:00
============================================================
Total Value: $10,000.00
Cash Balance: $6,000.00
Total Return: +0.00%
Active Positions: 1
  NVDA: 21.33 shares @ $187.62 (Unrealized: +$0.00, +0.00%)
============================================================
```

### Log File (`swing_paper_trader.log`):
- **Complete trade history**
- **All portfolio updates**
- **Entry/exit reasons**
- **Performance metrics**

## ⚙️ Configuration

### Edit `paper_trader.py` to customize:

```python
# Change monitored assets
TICKERS = ['NVDA', 'AMD', 'ETH-USD']  # Add/remove tickers

# Change initial balance
INITIAL_BALANCE = 10000  # Start with $10K

# Change leverage
LEVERAGE = 2.0  # Use 2x leverage (conservative)

# Change check frequency
CHECK_INTERVAL = 3600  # Check every hour (3600 seconds)
```

### Best Assets to Monitor:
- **NVDA** - Best performer (51% return, 6.22 Sharpe)
- **AMD** - Good performer (37% return, 3.63 Sharpe)
- **ETH-USD** - High volatility (82% return, 3.00 Sharpe)
- **COIN** - Crypto stock (28% return, 2.45 Sharpe)

## 📊 Expected Performance

### Based on Backtests:
- **Trades per week:** 0.2-1.3 per asset
- **Win rate:** 45-70% (varies by asset)
- **Average return:** 36% per 6 months
- **Sharpe ratio:** 2.8+ (excellent risk-adjusted returns)

### Realistic Expectations:
- **Paper trading:** Same as backtest (no costs)
- **Live trading:** 5-10% lower due to slippage/fees
- **Time required:** 1-2 minutes per day to check

## 🎯 What to Watch For

### Good Signs:
- ✅ **Consistent signals** - Getting 1-2 trades per week
- ✅ **High conviction entries** - 7+ points, 4+ signals
- ✅ **Winning trades** - 60%+ win rate
- ✅ **Good risk/reward** - 2:1 or better

### Warning Signs:
- ⚠️ **Too many trades** - More than 2 per day per asset
- ⚠️ **Low conviction** - Many 5-point entries
- ⚠️ **Low win rate** - Below 40%
- ⚠️ **Frequent stops** - Getting stopped out often

## 🛠️ Troubleshooting

### "No data for [ticker]"
- **Cause:** Market closed or data unavailable
- **Solution:** Wait for market hours (9:30 AM - 4:00 PM EST)

### "No signal detected"
- **Cause:** Strategy waiting for proper setup
- **Solution:** Normal - strategy is selective

### "Error fetching data"
- **Cause:** Internet connection or API issue
- **Solution:** Check connection, restart if needed

### High memory usage
- **Cause:** Long-running process
- **Solution:** Restart daily or weekly

## 📈 Performance Tracking

### Daily Check:
```bash
tail -50 swing_paper_trader.log
```

### Weekly Summary:
```bash
grep "TRADING SUMMARY" swing_paper_trader.log
```

### Trade Analysis:
```bash
grep "BUY\|SELL" swing_paper_trader.log
```

## 🎓 Tips for Success

### 1. **Run for at least 1 month**
- Strategy needs time to show patterns
- Don't judge on 1-2 trades

### 2. **Monitor during market hours**
- 9:30 AM - 4:00 PM EST
- Strategy works best with fresh data

### 3. **Check logs regularly**
- Look for consistent performance
- Watch for any errors

### 4. **Don't interfere**
- Let the strategy run automatically
- Don't manually close positions

### 5. **Track performance**
- Compare to buy-and-hold
- Monitor Sharpe ratio
- Watch win rate trends

## 🚨 Important Notes

### Paper Trading Only:
- **No real money at risk**
- **Perfect for testing strategy**
- **Learn how it works before going live**

### Market Hours:
- **US markets:** 9:30 AM - 4:00 PM EST
- **Crypto:** 24/7 (but strategy works best during US hours)
- **Weekends:** Limited activity

### Data Requirements:
- **Internet connection** needed
- **yfinance** for market data
- **4+ hours of data** for signals

## 🎯 Next Steps

### After 1 Month of Paper Trading:

1. **Analyze results:**
   - Did it beat buy-and-hold?
   - What was the win rate?
   - Any issues or errors?

2. **If successful:**
   - Consider live trading with small capital
   - Start with 10-20% of intended capital
   - Use 2x leverage (not 3x)

3. **If not successful:**
   - Check logs for errors
   - Try different assets
   - Adjust parameters

### Going Live:
- **Start small** - $1K-5K
- **Use 2x leverage** initially
- **Monitor closely** for first month
- **Scale up** if profitable

---

## 🚀 Ready to Start?

```bash
python paper_trader.py
```

**Let it run for a few days, then check the logs!**

The strategy is designed to be **patient and selective** - it may not trade every day, but when it does, it should be profitable.

Good luck! 🎯