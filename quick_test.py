"""
Quick test of the paper trader to make sure it works.
"""

from paper_trader import SwingPaperTrader
import time

def test_paper_trader():
    """Test the paper trader with a short run."""
    print("🧪 Testing Swing Paper Trader...")
    
    # Create trader
    trader = SwingPaperTrader(initial_balance=10000, leverage=2.0)
    
    # Test data fetching
    print("\n📊 Testing data fetching...")
    data = trader.get_live_data('NVDA', '1h', '2d')
    if data is not None:
        print(f"✅ NVDA data: {len(data)} bars")
        print(f"   Latest price: ${data['Close'].iloc[-1]:.2f}")
    else:
        print("❌ Failed to fetch NVDA data")
        return
    
    # Test 4h conversion
    print("\n🔄 Testing 4-hour conversion...")
    data_4h = trader.convert_to_4h(data)
    if data_4h is not None:
        print(f"✅ 4h data: {len(data_4h)} bars")
        print(f"   Latest 4h price: ${data_4h['Close'].iloc[-1]:.2f}")
    else:
        print("❌ Failed to convert to 4h")
        return
    
    # Test signal checking
    print("\n🎯 Testing signal detection...")
    signal, trade_data = trader.check_signals('NVDA', data_4h)
    if signal:
        print(f"✅ Signal detected: {signal}")
        print(f"   Trade data: {trade_data}")
    else:
        print("ℹ️  No signal detected (normal)")
    
    print("\n✅ Paper trader test completed successfully!")
    print("Ready to run: python paper_trader.py")

if __name__ == "__main__":
    test_paper_trader()