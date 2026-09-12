import pandas as pd
import numpy as np
import json
import os
import warnings
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

ASSETS = ['NSEI', 'NSEBANK', 'CNXPHARMA', 'CNXAUTO', 'CNXIT', 'CNXMETAL', 'CNXFMCG', 'CNXREALTY']
REGIMES = {0: "True_Bearish_Distribution", 1: "Choppy_or_Steady_Consolidation", 2: "High_Momentum_Bull_Breakout"}

def process_meta_learner():
    live_regimes = {}
    ensemble_history = []
    
    for asset in ASSETS:
        file_name = f"{asset}_base_preds.csv"
        if not os.path.exists(file_name):
            continue
            
        try:
            df = pd.read_csv(file_name, index_col=0, parse_dates=True)
            if df.empty:
                continue
                
            # Correctly sum the consensus for classes 0, 1, and 2
            c0 = np.sum(df.values[:, 0::3], axis=1)
            c1 = np.sum(df.values[:, 1::3], axis=1)
            c2 = np.sum(df.values[:, 2::3], axis=1)
            y_proxy = np.argmax(np.column_stack([c0, c1, c2]), axis=1)
            
            # Logistic regression requires at least 2 classes. Inject one if perfectly monotonic.
            if len(np.unique(y_proxy)) < 2:
                y_proxy[-1] = 1 if y_proxy[-1] != 1 else 0
                
            meta_model = LogisticRegression(class_weight='balanced', random_state=42)
            meta_model.fit(df.values, y_proxy)
            
            probs = meta_model.predict_proba(df.values)
            latest_probs = probs[-1]
            top_2_idx = np.argsort(latest_probs)[-2:]
            
            # Safe Knife-Edge Filter Extraction
            if len(top_2_idx) == 2:
                margin = abs(latest_probs[top_2_idx[1]] - latest_probs[top_2_idx[0]])
                if margin <= 0.05:
                    final_regime = 1 
                else:
                    final_regime = meta_model.classes_[top_2_idx[1]]
            else:
                final_regime = meta_model.classes_[top_2_idx[0]]
                
            live_regimes[asset] = {
                "regime": REGIMES[final_regime],
                "confidence": float(latest_probs[top_2_idx[-1]]),
                "date": df.index[-1].strftime("%Y-%m-%d")
            }
            
            historical_regimes = [REGIMES[meta_model.classes_[idx]] for idx in np.argmax(probs, axis=1)]
            hist_df = pd.DataFrame({
                'Date': df.index, 
                'Asset': asset, 
                'Regime': historical_regimes
            })
            ensemble_history.append(hist_df)
            
        except Exception as e:
            print(f"Error processing {asset}: {e}")
            
    try:
        with open("market_regimes.json", "w") as f:
            json.dump(live_regimes, f, indent=4)
            
        if ensemble_history:
            final_history = pd.concat(ensemble_history)
            final_history.to_csv("ensemble_history.csv", mode='a', header=not os.path.exists("ensemble_history.csv"))
        else:
            if not os.path.exists("ensemble_history.csv"):
                with open("ensemble_history.csv", "w") as f:
                    f.write("Date,Asset,Regime\n")
                    
    except Exception as e:
        print(f"Failed to save artifacts: {e}")
        with open("market_regimes.json", "w") as f: json.dump({}, f)

if __name__ == "__main__":
    process_meta_learner()
