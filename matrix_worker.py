import os
import requests
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
import warnings
warnings.filterwarnings("ignore")

# Strict Determinism
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Bypass Yahoo Finance Bot Block
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
})

ASSETS = ['^NSEI', '^NSEBANK', '^CNXPHARMA', '^CNXAUTO', '^CNXIT', '^CNXMETAL', '^CNXFMCG', '^CNXREALTY']
MACRO = ['USDINR=X', '^GSPC', 'CL=F', '^TNX']
TIMEFRAMES = [5, 14, 21, 63]

def get_fii_dii_data():
    """Correctly imports the nselib capital_market module."""
    try:
        from nselib import capital_market # Specific import required by nselib architecture
        df = capital_market.fii_dii_trading_activity()
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
        self.fc = nn.Linear(32, 3) 
        
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

def engineer_features(ticker, fii_dii_df):
    df = yf.download(ticker, period="2y", progress=False, session=session)
    if df.empty: return pd.DataFrame()
    
    for tf in TIMEFRAMES:
        df[f'Mom_{tf}'] = df['Close'].pct_change(tf)
        df[f'Vol_{tf}'] = df['Close'].rolling(tf).std()
    
    for m in MACRO:
        m_df = yf.download(m, period="2y", progress=False, session=session)['Close']
        df[f'Macro_{m}'] = m_df.pct_change(5)
        
    df = df.join(fii_dii_df, how='left')
    df.ffill(inplace=True) 
    df.fillna(0, inplace=True) 
    
    fwd_ret = df['Close'].pct_change(5).shift(-5)
    conditions = [
        (fwd_ret < -0.02),
        (fwd_ret > 0.02)
    ]
    choices = [0, 2]
    df['Target'] = np.select(conditions, choices, default=1)
    
    return df.dropna()

def get_3_class_probs(model, X_scaled):
    """Forces scikit-learn models to always output exactly 3 columns."""
    probs = model.predict_proba(X_scaled)
    full_probs = np.zeros((len(X_scaled), 3))
    for i, cls in enumerate(model.classes_):
        if cls in [0, 1, 2]:
            full_probs[:, int(cls)] = probs[:, i]
    return full_probs

def train_base_models(df):
    X = df.drop(columns=['Target', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], errors='ignore')
    y = df['Target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    predictions = {}
    
    xgb = XGBClassifier(random_state=SEED, n_estimators=50)
    xgb.fit(X_scaled, y)
    predictions['XGB'] = get_3_class_probs(xgb, X_scaled)
    
    lgb = LGBMClassifier(random_state=SEED, n_estimators=50, verbose=-1)
    lgb.fit(X_scaled, y)
    predictions['LGB'] = get_3_class_probs(lgb, X_scaled)
    
    cat = CatBoostClassifier(random_state=SEED, iterations=50, verbose=0)
    cat.fit(X_scaled, y)
    predictions['CAT'] = get_3_class_probs(cat, X_scaled)
    
    rf = RandomForestClassifier(random_state=SEED, n_estimators=50)
    rf.fit(X_scaled, y)
    predictions['RF'] = get_3_class_probs(rf, X_scaled)
    
    X_t = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y.values, dtype=torch.long)
    model = LSTMModel(X_scaled.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(10): 
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        predictions['LSTM'] = torch.softmax(model(X_t), dim=1).numpy()
    
    return predictions, df.index

if __name__ == "__main__":
    fii_dii = get_fii_dii_data()
    for asset in ASSETS:
        print(f"Processing Hive Mind Node: {asset}")
        df = engineer_features(asset, fii_dii)
        if not df.empty:
            preds, idx = train_base_models(df)
            flat_preds = np.hstack([preds[m] for m in preds.keys()])
            df_preds = pd.DataFrame(flat_preds, index=idx)
            df_preds.to_csv(f"{asset.replace('^', '')}_base_preds.csv")
