import yfinance as yf  # Import yfinance to get stock price
import pandas as pd

def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period='1y')
    return data

def add_technical_indicators(data):
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    std_dev = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (std_dev * 2)
    data['BB_Lower'] = data['BB_Middle'] - (std_dev * 2)
    data['Volume_MA_20'] = data['Volume'].rolling(window=20).mean()
    data['RSI'] = compute_rsi(data['Close'], 14)
    data.dropna(inplace=True)
    return data

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_features(data):
    data = add_technical_indicators(data)
    data['Return'] = data['Close'].pct_change()
    data['Direction'] = (data['Return'] > 0).astype(int)  # 1 if up, 0 if down
    data['Volume_Trend'] = (data['Volume'].diff() > 0).astype(int)  # 1 if increasing, 0 if decreasing
    data.dropna(inplace=True)
    return data[['SMA_5', 'SMA_10', 'EMA_12', 'EMA_26', 'MACD', 'Signal_Line', 
                  'BB_Upper', 'BB_Lower', 'Volume_MA_20', 'RSI', 'Close', 'Volume_Trend']], data['Direction']
