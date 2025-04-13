# S&P 500 Regime-Switching Model with Headline Sentiment and Macro Features

# ## 1. Setup and Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import yfinance as yf
import datetime
import requests
from textblob import TextBlob
from datetime import timedelta

# ## 2. Load Historical Market Indexes
tickers = ['^GSPC', '^IXIC', '^DJI', '^RUT']
data_dict = {}
for ticker in tickers:
    df = yf.download(ticker, start='2010-01-01', end=datetime.datetime.today().strftime('%Y-%m-%d'))[['Close']]
    df.columns = [ticker]
    data_dict[ticker] = df

market_data = pd.concat(data_dict.values(), axis=1)
market_data.dropna(inplace=True)

for ticker in tickers:
    market_data[f'{ticker}_Return'] = market_data[ticker].pct_change()

# ## 3. Load Macroeconomic Data
fred_tickers = {
    '10Y_Treasury_Yield': '^TNX',
    'Fed_Funds_Rate': 'FEDFUNDS'
}
for name, fred_ticker in fred_tickers.items():
    df = yf.download(fred_ticker, start='2010-01-01', end=datetime.datetime.today().strftime('%Y-%m-%d'))[['Close']]
    df = df.rename(columns={'Close': name})
    market_data = market_data.join(df[name], how='left')
market_data.fillna(method='ffill', inplace=True)

# ## 4. Load VIX as Sentiment Proxy
vix = yf.download('^VIX', start='2010-01-01', end=datetime.datetime.today().strftime('%Y-%m-%d'))[['Close']]
vix.columns = ['VIX']
vix = vix.dropna()
market_data = market_data.join(vix['VIX'], how='left')
market_data.fillna(method='ffill', inplace=True)

# ## 5. Headline-Based Sentiment Analysis using NewsAPI

# Set your NewsAPI key here
API_KEY = "YOUR_NEWSAPI_KEY"  # 🔐 Replace this with your actual key

# Function to fetch headlines for a specific date
def fetch_headlines(date, query="S&P 500"):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": date,
        "to": date,
        "language": "en",
        "sortBy": "relevancy",
        "apiKey": API_KEY,
        "pageSize": 5,
    }
    response = requests.get(url, params=params)
    data = response.json()
    return [article["title"] for article in data.get("articles", [])]

# Function to get sentiment scores for recent days
def get_sentiment_scores(start_days_ago=90):
    today = datetime.datetime.today()
    sentiment_data = []
    for i in range(start_days_ago):
        date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            headlines = fetch_headlines(date)
            sentiment = [TextBlob(h).sentiment.polarity for h in headlines]
            avg_sentiment = sum(sentiment) / len(sentiment) if sentiment else 0.0
        except Exception as e:
            avg_sentiment = None
        sentiment_data.append((date, avg_sentiment))
    return pd.DataFrame(sentiment_data, columns=["Date", "Headline_Sentiment"])

sentiment_df = get_sentiment_scores()
sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
sentiment_df = sentiment_df.sort_values('Date')
sentiment_df.set_index('Date', inplace=True)

# 📈 Visualize Headline Sentiment Trends
plt.figure(figsize=(14, 5))
sentiment_df['Headline_Sentiment'].plot(title='Headline Sentiment Over Time', grid=True)
plt.xlabel('Date')
plt.ylabel('Sentiment Polarity')
plt.show()

# Merge into main data
market_data = market_data.merge(sentiment_df, left_index=True, right_index=True, how='left')
market_data['Headline_Sentiment'].fillna(method='ffill', inplace=True)

# ## 6. Feature Engineering
window = 10
market_data['Momentum'] = market_data['^GSPC'] - market_data['^GSPC'].shift(window)
market_data['Volatility'] = market_data['^GSPC_Return'].rolling(window).std()
market_data['RollingMean'] = market_data['^GSPC'].rolling(window).mean()
market_data['RollingStd'] = market_data['^GSPC'].rolling(window).std()
market_data['Lag1'] = market_data['^GSPC_Return'].shift(1)
market_data['Lag2'] = market_data['^GSPC_Return'].shift(2)
market_data.dropna(inplace=True)

# ## 7. Regime Detection using KMeans
features = ['RollingMean', 'RollingStd', 'Momentum', 'Volatility', 'Lag1', 'Lag2',
            '^IXIC_Return', '^DJI_Return', '^RUT_Return',
            '10Y_Treasury_Yield', 'Fed_Funds_Rate', 'VIX', 'Headline_Sentiment']
X_regime = StandardScaler().fit_transform(market_data[features])
kmeans = KMeans(n_clusters=3, random_state=42)
market_data['Regime'] = kmeans.fit_predict(X_regime)

# ## 8. Visualize Regimes
plt.figure(figsize=(14,6))
for regime in sorted(market_data['Regime'].unique()):
    subset = market_data[market_data['Regime'] == regime]
    plt.plot(subset.index, subset['^GSPC'], label=f'Regime {regime}')
plt.legend()
plt.title('S&P 500 Price by Regime')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# ## 9. Train Regime-Specific Models
models = {}
target_col = '^GSPC_Return'
train_features = features
tscv = TimeSeriesSplit(n_splits=5)

for regime in sorted(market_data['Regime'].unique()):
    df_regime = market_data[market_data['Regime'] == regime]
    X = df_regime[train_features]
    y = df_regime[target_col]

    print(f"Training model for Regime {regime}...")
    regime_preds = pd.Series(index=df_regime.index, dtype='float64')

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        regime_preds.iloc[test_idx] = preds

    market_data.loc[df_regime.index, f'Preds_Regime{regime}'] = regime_preds
    models[regime] = model

# ## 10. Evaluate
for regime in sorted(market_data['Regime'].unique()):
    actual = market_data.loc[market_data['Regime'] == regime, target_col]
    pred = market_data.loc[market_data['Regime'] == regime, f'Preds_Regime{regime}']
    rmse = np.sqrt(mean_squared_error(actual, pred))
    print(f"RMSE for Regime {regime}: {rmse:.5f}")

# ## 11. Predict Next Few Days
latest_data = market_data.iloc[-1:]
latest_features = latest_data[train_features]
regime_today = int(latest_data['Regime'].values[0])
model_today = models[regime_today]
predictions = []

print(f"Current regime: {regime_today}")

for day in range(1, 6):
    pred_return = model_today.predict(latest_features)[0]
    predictions.append(pred_return)
    new_price = latest_data['^GSPC'].values[-1] * (1 + pred_return)

    new_row = latest_data.copy()
    new_row.index = [latest_data.index[-1] + pd.Timedelta(days=1)]
    new_row['^GSPC'] = new_price
    new_row['^GSPC_Return'] = pred_return
    new_row['Lag2'] = new_row['Lag1']
    new_row['Lag1'] = pred_return
    latest_data = pd.concat([latest_data, new_row])
    latest_features = new_row[train_features]

print("Predicted returns for next 5 days:", predictions)

# ## 12. Visualization
plt.figure(figsize=(14,6))
for regime in sorted(market_data['Regime'].unique()):
    plt.plot(market_data.loc[market_data['Regime'] == regime].index,
             market_data.loc[market_data['Regime'] == regime, f'Preds_Regime{regime}'],
             label=f'Regime {regime} Prediction')
plt.plot(market_data.index, market_data['^GSPC_Return'], color='black', alpha=0.3, label='Actual Return')
plt.legend()
plt.title('Predicted Returns by Regime vs Actual')
plt.show()

