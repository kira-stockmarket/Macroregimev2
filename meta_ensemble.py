import pandas as pd
import numpy as np
import json
import os
from sklearn.linear_model import LogisticRegression

ASSETS = ['NSEI', 'NSEBANK', 'CNXPHARMA', 'CNXAUTO', 'CNXIT', 'CNXMETAL', 'CNXFMCG', 'CNXREALTY']
REGIMES = {0: "True_Bearish_Distribution", 1: "Choppy_or_Steady_Consolidation", 2: "High_Momentum_Bull_Breakout"}

def process_meta_learner():
    live_regimes = {}
    ensemble_history = []
    
    for asset in ASSETS:
        file_name = f"{asset}_base_preds.csv"
        if not os.path.exists(file_name):
            print(f"File {file_name} not found. Skipping.")
            continue
            
        try:
            df = pd.read_csv(file_name, index_col=0, parse_dates=True)
            if df.empty:
                continue
                
            # Create a proxy target based on consensus of the models
            # Models output 3 probabilities each. We sum them up to find consensus.
            y_proxy = np.argmax(df.values[:, 0::3] + df.values[:, 1::3] + df.values[:, 2::3], axis=1)
            
            meta_model = LogisticRegression(class_weight='balanced', random_state=42)
            meta_model.fit(df.values, y_proxy)
            
            probs = meta_model.predict_proba(df.values)
            
            # Knife-Edge Filter logic on the latest data point
            latest_probs = probs[-1]
            top_2_idx = np.argsort(latest_probs)[-2:]
            margin = abs(latest_probs[top_2_idx[1]] - latest_probs[top_2_idx[0]])
            
            if margin <= 0.05:
                final_regime = 1 # Force Choppy on ambiguity
            else:
                final_regime = int(top_2_idx[1])
                
            live_regimes[asset] = {
                "regime": REGIMES[final_regime],
                "confidence": float(latest_probs[top_2_idx[1]]),
                "date": df.index[-1].strftime("%Y-%m-%d")
            }
            
            # Append to history
            hist_df = pd.DataFrame({
                'Date': df.index, 
                'Asset': asset, 
                'Regime': [REGIMES[r] for r in np.argmax(probs, axis=1)]
            })
            ensemble_history.append(hist_df)
            
        except Exception as e:
            print(f"Error processing {asset}: {e}")
            
    # Safely save outputs to ensure JSON is always created
    try:
        # Write valid JSON even if empty
        with open("market_regimes.json", "w") as f:
            json.dump(live_regimes, f, indent=4)
            
        if ensemble_history:
            final_history = pd.concat(ensemble_history)
            final_history.to_csv("ensemble_history.csv", mode='a', header=not os.path.exists("ensemble_history.csv"))
        else:
            print("No history to append today.")
            # Ensure history file exists so backtester doesn't crash later
            if not os.path.exists("ensemble_history.csv"):
                with open("ensemble_history.csv", "w") as f:
                    f.write("Date,Asset,Regime\n")
                    
    except Exception as e:
        print(f"Failed to save artifacts: {e}")
        # Absolute fallback: write valid empty JSON
        with open("market_regimes.json", "w") as f:
            json.dump({}, f)
        if not os.path.exists("ensemble_history.csv"):
            with open("ensemble_history.csv", "w") as f:
                f.write("Date,Asset,Regime\n")

if __name__ == "__main__":
    process_meta_learner()
