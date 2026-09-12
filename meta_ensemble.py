import pandas as pd
import numpy as np
import json
import os
from sklearn.linear_model import LogisticRegression
from datetime import datetime

ASSETS = ['NSEI', 'NSEBANK', 'CNXPHARMA', 'CNXAUTO', 'CNXIT', 'CNXMETAL', 'CNXFMCG', 'CNXREALTY']
REGIMES = {0: "True_Bearish_Distribution", 1: "Choppy_or_Steady_Consolidation", 2: "High_Momentum_Bull_Breakout"}

def process_meta_learner():
    live_regimes = {}
    ensemble_history = []
    
    for asset in ASSETS:
        try:
            df = pd.read_csv(f"{asset}_base_preds.csv", index_col=0, parse_dates=True)
            
            # Create a proxy target for meta-learner based on consensus
            y_proxy = np.argmax(df.iloc[:, [0,3,6,9,12]].values, axis=1) # Target proxy
            
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
            hist_df = pd.DataFrame({'Date': df.index, 'Asset': asset, 'Regime': [REGIMES[r] for r in np.argmax(probs, axis=1)]})
            ensemble_history.append(hist_df)
            
        except Exception as e:
            print(f"Error processing {asset}: {e}")
            
    # Save outputs
    try:
        with open("market_regimes.json", "w") as f:
            json.dump(live_regimes, f, indent=4)
            
        final_history = pd.concat(ensemble_history)
        final_history.to_csv("ensemble_history.csv", mode='a', header=not os.path.exists("ensemble_history.csv"))
    except Exception as e:
        print(f"Failed to save artifacts: {e}")
        # Fallback empty files
        open("market_regimes.json", "w").close()
        open("ensemble_history.csv", "w").close()

if __name__ == "__main__":
    process_meta_learner()
