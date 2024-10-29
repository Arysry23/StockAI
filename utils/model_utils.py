import yfinance as yf
import pandas as pd
import os
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from .data_utils import fetch_stock_data, prepare_features

DATABASE_FILE = "predictions.db"

# Create a SQLite database and a table for predictions if it doesn't exist
def create_db():
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            date TEXT,
            predicted_direction INTEGER,
            actual_direction INTEGER,
            correct INTEGER
        )
    ''')
    conn.commit()
    conn.close()

create_db()

def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def generate_recommendation(symbol):
    stock_info = yf.Ticker(symbol)
    try:
        current_data = stock_info.history(period='1d')
        current_price = current_data['Close'].iloc[0]
        current_volume = current_data['Volume'].iloc[0]
    except (IndexError, ValueError):
        current_price = None
        current_volume = None

    data = fetch_stock_data(symbol)
    if data.empty:
        return f"No data available for {symbol}.", current_price, current_volume, None

    X, y = prepare_features(data)
    if X.empty or y.empty:
        return f"Insufficient data for {symbol}.", current_price, current_volume, None

    model = train_model(X, y)
    probability_uptrend = model.predict_proba([X.iloc[-1]])[0][1] * 100
    recommendation = "Buy" if probability_uptrend > 50 else "Do not buy"

    volume_trend = "Increasing" if data['Volume_Trend'].iloc[-1] == 1 else "Decreasing"

    store_prediction(symbol, recommendation, current_price, current_volume)

    return f"{symbol}: {recommendation} (Probability of uptrend: {probability_uptrend:.2f}%)", current_price, current_volume, volume_trend

def store_prediction(symbol, prediction, current_price, current_volume):
    # Get the actual result based on last known data
    actual_direction = get_actual_direction(symbol)
    
    # Check if the prediction for the same day already exists
    existing_prediction = check_existing_prediction(symbol)
    
    if existing_prediction:
        print(f"Prediction already exists for {symbol} on {pd.Timestamp.now().date()}. Skipping...")
        return  # Skip storing if it already exists
    
    correct = 1 if prediction.startswith("Buy") and actual_direction == 1 else 0

    # Append prediction to the database
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (date, symbol, predicted_direction, actual_direction, correct)
        VALUES (?, ?, ?, ?, ?)
    ''', (pd.Timestamp.now().date(), symbol, prediction, actual_direction, correct))
    conn.commit()
    conn.close()

def check_existing_prediction(symbol):
    # Check for existing prediction for the same symbol and date
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT COUNT(*) FROM predictions 
        WHERE symbol = ? AND date = ?
    ''', (symbol, pd.Timestamp.now().date()))
    exists = cursor.fetchone()[0] > 0
    conn.close()
    return exists


def get_actual_direction(symbol):
    data = fetch_stock_data(symbol)

    if data.shape[0] < 2:  # Check if there's enough data
        print(f"Not enough data for {symbol}.")
        return None
    
    last_close = data['Close'].iloc[-1]
    second_last_close = data['Close'].iloc[-2]

    return 1 if last_close > second_last_close else 0

def calculate_accuracy(symbol):
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    c.execute('SELECT * FROM predictions WHERE symbol = ?', (symbol,))
    symbol_history = c.fetchall()
    conn.close()

    if not symbol_history:
        return 0.0  # No history for this symbol

    correct_predictions = sum(row[4] for row in symbol_history)  # Index 4 is 'correct'
    total_predictions = len(symbol_history)
    
    return (correct_predictions / total_predictions) * 100  # Return accuracy as a percentage
