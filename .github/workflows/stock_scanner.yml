import yfinance as yf
import pandas as pd
import json
import os
from datetime import datetime

# Define a sample universe mapped to our sectors
STOCK_UNIVERSE = {
    'NSEBANK': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS'],
    'CNXIT': ['TCS.NS', 'INFY.NS', 'HCLTECH.NS'],
    'CNXAUTO': ['TATAMOTORS.NS', 'M&M.NS', 'MARUTI.NS']
}

def scan_for_breakouts():
    signals = {}
    
    # Load market regimes to filter our scans
    try:
        with open("market_regimes.json", "r") as f:
            regimes = json.load(f)
    except FileNotFoundError:
        regimes = {}

    for sector, tickers in STOCK_UNIVERSE.items():
        # Only look for long breakouts if the sector isn't in a Bearish distribution
        sector_regime = regimes.get(sector, {}).get("regime", "Choppy_or_Steady_Consolidation")
        if sector_regime == "True_Bearish_Distribution":
            signals[sector] = {"status": "Halted", "reason": "Bearish Sector Regime", "breakouts": []}
            continue

        breakouts = []
        for ticker in tickers:
            try:
                # Fetch last 30 days of data
                df = yf.download(ticker, period="1mo", progress=False)
                if df.empty or len(df) < 21:
                    continue
                
                current_close = df['Close'].iloc[-1].item()
                prev_high_20 = df['High'].iloc[-21:-1].max().item()
                current_vol = df['Volume'].iloc[-1].item()
                avg_vol_20 = df['Volume'].iloc[-21:-1].mean().item()

                # Breakout Condition: Close > 20-Day High AND Volume > 1.5x 20-Day Avg
                if current_close > prev_high_20 and current_vol > (1.5 * avg_vol_20):
                    breakouts.append({
                        "ticker": ticker,
                        "close": round(current_close, 2),
                        "breakout_level": round(prev_high_20, 2)
                    })
            except Exception as e:
                print(f"Error scanning {ticker}: {e}")

        signals[sector] = {
            "status": "Active",
            "sector_regime": sector_regime,
            "breakouts": breakouts,
            "scan_date": datetime.now().strftime("%Y-%m-%d")
        }

    # Save to JSON for the dashboard
    with open("sector_signals.json", "w") as f:
        json.dump(signals, f, indent=4)

if __name__ == "__main__":
    scan_for_breakouts()
