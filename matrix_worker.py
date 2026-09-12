import os
import yfinance as yf
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import optuna
import nselib
import warnings
warnings.filterwarnings("ignore")

# Strict Determinism
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ASSETS = ['^NSEI', '^NSEBANK', '^CNXPHARMA', '^CNXAUTO', '^CNXIT', '^CNXMETAL', '^CNXFMCG', '^CNXREALTY']
MACRO = ['USDINR=X', '^GSPC', 'CL=F', '^TNX']
TIMEFRAMES = [5, 14, 21, 63]

def get_fii_dii_data():
    """Attempt to fetch live FII/DII data; fallback to mock data on CI/CD failure."""
    try:
        # nselib implementation for FII/DII derivatives/cash flow
        df = nselib.capital_market.fii_dii_trading_activity()
        df['Date'] = pd.to_datetime(df['Date'])
        return df.set_index('Date')
    except Exception as e:
        print(f"nselib fetch failed: {e}. Generating neutral proxy data.")
        dates = pd.date_range(end=pd.Timestamp.today(), periods=500)
        return pd.DataFrame({'FII_Net': np.random.randn(500)*100, 'DII_Net': np.random.randn(500)*100}, index=dates)

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 32, batch_first=True)
        self.fc = nn.Linear(32, 3) # 3 Regimes
        
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

def engineer_features(ticker):
    df = yf.download(ticker, period="2y", progress=False)
    if df.empty: return pd.DataFrame()
    
    # Base Features
    for tf in TIMEFRAMES:
        df[f'Mom_{tf}'] = df['Close'].pct_change(tf)
        df[f'Vol_{tf}'] = df['Close'].rolling(tf).std()
    
    # Macro Data
    for m in MACRO:
        m_df = yf.download(m, period="2y", progress=False)['Close']
        df[f'Macro_{m}'] = m_df.pct_change(5)
        
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    
    # Regime Target Logic: 0 = Bearish, 1 = Choppy, 2 = Bullish
    fwd_ret = df['Close'].pct_change(5).shift(-5)
    conditions = [
        (fwd_ret < -0.02),
        (fwd_ret > 0.02)
    ]
    choices = [0, 2]
    df['Target'] = np.select(conditions, choices, default=1)
    
    return df.dropna()

def train_base_models(df):
    X = df.drop(columns=['Target', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], errors='ignore')
    y = df['Target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    predictions = {}
    
    # 1. XGBoost
    xgb = XGBClassifier(random_state=SEED, n_estimators=50)
    xgb.fit(X_scaled, y)
    predictions['XGB'] = xgb.predict_proba(X_scaled)
    
    # 2. LightGBM
    lgb = LGBMClassifier(random_state=SEED, n_estimators=50, verbose=-1)
    lgb.fit(X_scaled, y)
    predictions['LGB'] = lgb.predict_proba(X_scaled)
    
    # 3. CatBoost
    cat = CatBoostClassifier(random_state=SEED, iterations=50, verbose=0)
    cat.fit(X_scaled, y)
    predictions['CAT'] = cat.predict_proba(X_scaled)
    
    # 4. Random Forest
    rf = RandomForestClassifier(random_state=SEED, n_estimators=50)
    rf.fit(X_scaled, y)
    predictions['RF'] = rf.predict_proba(X_scaled)
    
    # 5. PyTorch LSTM (Simplified for CI/CD)
    X_t = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y.values, dtype=torch.long)
    model = LSTMModel(X_scaled.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(10): # Fast train for CI/CD
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        lstm_preds = torch.softmax(model(X_t), dim=1).numpy()
    predictions['LSTM'] = lstm_preds
    
    return predictions, df.index

if __name__ == "__main__":
    fii_dii = get_fii_dii_data()
    all_preds = {}
    
    for asset in ASSETS:
        print(f"Processing Hive Mind Node: {asset}")
        df = engineer_features(asset)
        if not df.empty:
            preds, idx = train_base_models(df)
            # Flatten predictions for meta-learner
            flat_preds = np.hstack([preds[m] for m in preds.keys()])
            df_preds = pd.DataFrame(flat_preds, index=idx)
            df_preds.to_csv(f"{asset.replace('^', '')}_base_preds.csv")
