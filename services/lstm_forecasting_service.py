import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout
from keras.optimizers import Adam
from utils.metrics_utils import evaluate_forecast
from utils.plot_utils import plot_forecast
import os
import joblib
from pathlib import Path

def create_sequences(data, window_size):
    X, y = ([], [])
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, :])
        y.append(data[i + window_size, 0])
    return (np.array(X), np.array(y))

class LSTMBidirectionalForecaster:

    def __init__(self, window_size=12, epochs=100, batch_size=8, model_name='default'):
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler()
        self.model_name = model_name
        self.fitted = False
        self.models_dir = Path('models/lstm')
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self):
        return self.models_dir / f'{self.model_name}.pkl'

    def fit(self, train_df):
        model_path = self.get_model_path()
        if model_path.exists():
            self.load_model()
            return
        train_df = train_df.copy()
        train_df['month'] = pd.to_datetime(train_df['ds']).dt.month
        y_train_scaled = self.scaler.fit_transform(train_df[['y']]).flatten()
        train_features = np.column_stack([y_train_scaled, train_df['month'].values / 12.0])
        X_train, y_train = create_sequences(train_features, self.window_size)
        self.model = Sequential([Bidirectional(LSTM(64, return_sequences=True), input_shape=(self.window_size, 2)), Dropout(0.2), Bidirectional(LSTM(32)), Dropout(0.2), Dense(1)])
        self.model.compile(optimizer=Adam(0.001), loss='mse')
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.fitted = True
        self.save_model()

    def save_model(self):
        model_path = self.get_model_path()
        joblib.dump({'model_weights': self.model.get_weights(), 'scaler_params': self.scaler.__dict__, 'window_size': self.window_size, 'fitted': True}, model_path)

    def load_model(self):
        model_path = self.get_model_path()
        if model_path.exists():
            saved_data = joblib.load(model_path)
            self.model = Sequential([Bidirectional(LSTM(64, return_sequences=True), input_shape=(saved_data['window_size'], 2)), Dropout(0.2), Bidirectional(LSTM(32)), Dropout(0.2), Dense(1)])
            self.model.compile(optimizer=Adam(0.001), loss='mse')
            self.model.set_weights(saved_data['model_weights'])
            self.scaler.__dict__ = saved_data['scaler_params']
            self.window_size = saved_data['window_size']
            self.fitted = saved_data['fitted']
            return True
        return False

    def predict(self, train_df, n_steps):
        if not self.fitted:
            raise RuntimeError('Modèle non entraîné.')
        train_df = train_df.copy()
        train_df['month'] = pd.to_datetime(train_df['ds']).dt.month
        y_train_scaled = self.scaler.transform(train_df[['y']]).flatten()
        train_features = np.column_stack([y_train_scaled, train_df['month'].values / 12.0])
        last_window = train_features[-self.window_size:]
        forecast_scaled = []
        for _ in range(n_steps):
            X_input = np.array(last_window).reshape(1, self.window_size, -1)
            y_pred = self.model.predict(X_input, verbose=0)[0, 0]
            forecast_scaled.append(y_pred)
            next_month = (int(last_window[-1, 1] * 12) % 12 + 1) / 12.0
            next_input = np.array([y_pred, next_month])
            last_window = np.vstack([last_window[1:], next_input])
        forecast_values = self.scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
        return forecast_values

def split_data(df, val_start, test_start):
    train = df[df['ds'] < val_start]
    val = df[(df['ds'] >= val_start) & (df['ds'] < test_start)]
    test = df[df['ds'] >= test_start]
    return (train, val, test)

def forecast_by_product_lstm(df, product_id, val_start='2015-07-01', test_start='2016-01-01'):
    product_df = df[df['id'] == product_id][['ds', 'sales']].rename(columns={'sales': 'y'})
    train_df, val_df, test_df = split_data(product_df, val_start, test_start)
    full_truth = pd.concat([train_df, val_df, test_df])
    lstm_model = LSTMBidirectionalForecaster(window_size=12, epochs=100, batch_size=4, model_name=f'product_{product_id}')
    lstm_model.fit(train_df)
    n_steps = len(val_df) + len(test_df)
    forecast_values = lstm_model.predict(train_df, n_steps)
    forecast_dates = full_truth['ds'][-n_steps:]
    forecast_df = pd.DataFrame({'ds': forecast_dates, 'y': full_truth['y'][-n_steps:].values, 'yhat': forecast_values})
    scores = evaluate_forecast(forecast_df)
    fig = plot_forecast(data=full_truth, forecast=forecast_df, train_end=train_df['ds'].max(), title='Prévision LSTM Bidirectionnel', label=f'Produit {product_id}', model='LSTM Bidirectionnel')
    return (forecast_df, scores, fig)

def forecast_by_category_lstm(df, category_id, val_start='2015-07-01', test_start='2016-01-01', category_column='cat_id'):
    category_df = df[df[category_column] == category_id].groupby('ds', as_index=False)['sales'].sum().rename(columns={'sales': 'y'})
    train_df, val_df, test_df = split_data(category_df, val_start, test_start)
    full_truth = pd.concat([train_df, val_df, test_df])
    lstm_model = LSTMBidirectionalForecaster(window_size=12, epochs=100, batch_size=4, model_name=f'category_{category_id}')
    lstm_model.fit(train_df)
    n_steps = len(val_df) + len(test_df)
    forecast_values = lstm_model.predict(train_df, n_steps)
    forecast_dates = full_truth['ds'][-n_steps:]
    forecast_df = pd.DataFrame({'ds': forecast_dates, 'y': full_truth['y'][-n_steps:].values, 'yhat': forecast_values})
    scores = evaluate_forecast(forecast_df)
    fig = plot_forecast(data=full_truth, forecast=forecast_df, train_end=train_df['ds'].max(), title='Prévision LSTM Bidirectionnel', label=f'Catégorie {category_id}', model='LSTM Bidirectionnel')
    return (forecast_df, scores, fig)

def forecast_by_store_lstm(df, store_id, val_start='2015-07-01', test_start='2016-01-01'):
    store_df = df[df['store_id'] == store_id].groupby('ds')['sales'].sum().reset_index().rename(columns={'sales': 'y'})
    train_df, val_df, test_df = split_data(store_df, val_start, test_start)
    full_truth = pd.concat([train_df, val_df, test_df])
    lstm_model = LSTMBidirectionalForecaster(window_size=12, epochs=100, batch_size=4, model_name=f'store_{store_id}')
    lstm_model.fit(train_df)
    n_steps = len(val_df) + len(test_df)
    forecast_values = lstm_model.predict(train_df, n_steps)
    forecast_dates = full_truth['ds'][-n_steps:]
    forecast_df = pd.DataFrame({'ds': forecast_dates, 'y': full_truth['y'][-n_steps:].values, 'yhat': forecast_values})
    scores = evaluate_forecast(forecast_df)
    fig = plot_forecast(data=full_truth, forecast=forecast_df, train_end=train_df['ds'].max(), title='Prévision LSTM Bidirectionnel', label=f'Magasin {store_id}', model='LSTM Bidirectionnel')
    return (forecast_df, scores, fig)

def run_lstm_forecast(df, granularity='product', value=None, val_start='2015-07-01', test_start='2016-01-01'):
    if granularity == 'product' and value:
        return forecast_by_product_lstm(df, value, val_start, test_start)
    if granularity == 'category' and value:
        granularity_column = 'product_category' if 'product_category' in df.columns else 'cat_id'
        return forecast_by_category_lstm(df, value, val_start, test_start, category_column=granularity_column)
    if granularity == 'store' and value:
        return forecast_by_store_lstm(df, value, val_start, test_start)
    raise ValueError('Granularité invalide ou identifiant manquant.')