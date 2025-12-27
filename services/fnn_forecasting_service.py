import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import KBinsDiscretizer
from utils.metrics_utils import evaluate_forecast
from utils.plot_utils import plot_forecast
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.neural_network import MLPRegressor

def create_lag_features(df, lags=12):
    df = df.copy()
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['y'].shift(i)
    df['month'] = pd.to_datetime(df['ds']).dt.month
    return df.dropna().reset_index(drop=True)

def fuzzy_transform(X, n_bins=5):
    transformer = KBinsDiscretizer(n_bins=n_bins, encode='onehot-dense', strategy='quantile')
    return transformer.fit_transform(X)

def split_data(df, val_start, test_start):
    train = df[df['ds'] < val_start]
    val = df[(df['ds'] >= val_start) & (df['ds'] < test_start)]
    test = df[df['ds'] >= test_start]
    return (train, val, test)

def train_fnn_forecast(train_df, val_df, test_df, target_col='y'):
    features = [col for col in train_df.columns if col not in ['ds', target_col]]
    scaler_X = StandardScaler()
    X_train_raw = train_df[features]
    X_val_raw = val_df[features]
    X_test_raw = test_df[features]
    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    X_val_scaled = scaler_X.transform(X_val_raw)
    X_test_scaled = scaler_X.transform(X_test_raw)
    transformer = KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile')
    X_train = transformer.fit_transform(X_train_scaled)
    X_val = transformer.transform(X_val_scaled)
    X_test = transformer.transform(X_test_scaled)
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(train_df[[target_col]]).ravel()
    y_val = scaler_y.transform(val_df[[target_col]]).ravel()
    y_test = scaler_y.transform(test_df[[target_col]]).ravel()
    model = MLPRegressor(hidden_layer_sizes=(50,), activation='relu', solver='adam', random_state=42, max_iter=500)
    model.fit(X_train, y_train)
    yhat_val = scaler_y.inverse_transform(model.predict(X_val).reshape(-1, 1)).ravel()
    yhat_test = scaler_y.inverse_transform(model.predict(X_test).reshape(-1, 1)).ravel()
    forecast_df = pd.concat([val_df[['ds', 'y']].assign(yhat=yhat_val), test_df[['ds', 'y']].assign(yhat=yhat_test)])
    return (forecast_df, model)

def forecast_by_product_fnn(df, product_id, val_start, test_start):
    product_df = df[df['id'] == product_id][['ds', 'sales']].rename(columns={'sales': 'y'})
    product_df_lagged = create_lag_features(product_df)
    train, val, test = split_data(product_df_lagged, val_start, test_start)
    full_truth = pd.concat([train, val, test])
    forecast_df, model = train_fnn_forecast(train, val, test)
    scores = evaluate_forecast(forecast_df)
    fig = plot_forecast(full_truth, forecast_df, train['ds'].max(), 'Prévision', f'Produit {product_id}', 'FNN')
    return (forecast_df, scores, fig)

def forecast_by_category_fnn(df, category_id, val_start, test_start, category_column='cat_id'):
    category_df = df[df[category_column] == category_id].groupby('ds', as_index=False)['sales'].sum().rename(columns={'sales': 'y'})
    category_df_lagged = create_lag_features(category_df)
    train, val, test = split_data(category_df_lagged, val_start, test_start)
    full_truth = pd.concat([train, val, test])
    forecast_df, model = train_fnn_forecast(train, val, test)
    scores = evaluate_forecast(forecast_df)
    fig = plot_forecast(full_truth, forecast_df, train['ds'].max(), 'Prévision', f'Catégorie {category_id}', 'FNN')
    return (forecast_df, scores, fig)

def forecast_by_store_fnn(df, store_id, val_start, test_start):
    store_df = df[df['store_id'] == store_id].groupby('ds', as_index=False)['sales'].sum().rename(columns={'sales': 'y'})
    store_df_lagged = create_lag_features(store_df)
    train, val, test = split_data(store_df_lagged, val_start, test_start)
    full_truth = pd.concat([train, val, test])
    forecast_df, model = train_fnn_forecast(train, val, test)
    scores = evaluate_forecast(forecast_df)
    fig = plot_forecast(full_truth, forecast_df, train['ds'].max(), 'Prévision', f'Magasin {store_id}', 'FNN')
    return (forecast_df, scores, fig)

def run_fnn_forecast(df, granularity='product', value=None, val_start='2015-07-01', test_start='2016-01-01'):
    if granularity == 'product' and value:
        return forecast_by_product_fnn(df, value, val_start, test_start)
    if granularity == 'category' and value:
        granularity_column = 'product_category' if 'product_category' in df.columns else 'cat_id'
        return forecast_by_category_fnn(df, value, val_start, test_start, granularity_column)
    if granularity == 'store' and value:
        return forecast_by_store_fnn(df, value, val_start, test_start)
    raise ValueError('Granularité invalide ou identifiant manquant.')