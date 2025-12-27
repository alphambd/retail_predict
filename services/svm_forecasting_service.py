import os
import json
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from utils.metrics_utils import evaluate_forecast
from utils.plot_utils import plot_forecast
CACHE_DIR = 'models_cache/svm'
os.makedirs(CACHE_DIR, exist_ok=True)

def create_lag_features(df, lags=6):
    df = df.copy()
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['y'].shift(i)
    df['month'] = pd.to_datetime(df['ds']).dt.month
    return df.dropna().reset_index(drop=True)

def split_data(df, val_start, test_start):
    train = df[df['ds'] < val_start]
    val = df[(df['ds'] >= val_start) & (df['ds'] < test_start)]
    test = df[df['ds'] >= test_start]
    return (train, val, test)

def get_cache_filename(granularity, value):
    return os.path.join(CACHE_DIR, f'{granularity}_{value}.json')

def load_or_search_best_params(X, y, granularity, value):
    cache_file = get_cache_filename(granularity, value)
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            best_params = json.load(f)
            return best_params
            return best_params
    param_grid = {'C': [0.1, 1, 10], 'epsilon': [0.01, 0.1, 0.5], 'kernel': ['rbf', 'linear']}
    model = SVR()
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    with open(cache_file, 'w') as f:
        json.dump(best_params, f)
        return best_params
        return best_params

def train_svm_forecast(train_df, val_df, test_df, granularity, value, target_col='y'):
    features = [col for col in train_df.columns if col not in ['ds', target_col]]
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(train_df[features])
    X_val = scaler_X.transform(val_df[features])
    X_test = scaler_X.transform(test_df[features])
    y_train = scaler_y.fit_transform(train_df[[target_col]]).ravel()
    best_params = load_or_search_best_params(X_train, y_train, granularity, value)
    model = SVR(**best_params)
    model.fit(X_train, y_train)
    yhat_val = scaler_y.inverse_transform(model.predict(X_val).reshape(-1, 1)).ravel()
    yhat_test = scaler_y.inverse_transform(model.predict(X_test).reshape(-1, 1)).ravel()
    forecast_df = pd.concat([val_df[['ds', 'y']].assign(yhat=yhat_val), test_df[['ds', 'y']].assign(yhat=yhat_test)])
    return (forecast_df, model)
pass

def forecast_by_svm(df, value, val_start, test_start, granularity_column):
    if granularity_column == 'id' or granularity_column == 'product_id':
        target_df = df[df[granularity_column] == value][['ds', 'sales']].rename(columns={'sales': 'y'})
        label = f'Produit {value}'
    else:
        target_df = df[df[granularity_column] == value].groupby('ds', as_index=False)['sales'].sum().rename(columns={'sales': 'y'})
        label = f'{granularity_column.capitalize()} {value}'
    lagged_df = create_lag_features(target_df)
    train, val, test = split_data(lagged_df, val_start, test_start)
    full_truth = pd.concat([train, val, test])
    forecast_df, model = train_svm_forecast(train, val, test, granularity_column, value)
    scores = evaluate_forecast(forecast_df)
    fig = plot_forecast(full_truth, forecast_df, train['ds'].max(), 'Prévision', label, 'SVM')
    return (forecast_df, scores, fig)
pass

def run_svm_forecast(df, granularity='product', value=None, val_start='2015-07-01', test_start='2016-01-01'):
    if granularity == 'product':
        granularity_column = 'id' if 'id' in df.columns else 'product_id'
    elif granularity == 'category':
        granularity_column = 'cat_id' if 'cat_id' in df.columns else 'product_category'
    elif granularity == 'store':
        granularity_column = 'store_id' if 'store_id' in df.columns else 'store_name'
    else:
        raise ValueError('Granularité invalide.')
    if value is None:
        raise ValueError("Valeur d'identifiant manquante pour la granularité choisie.")
    return forecast_by_svm(df, value, val_start, test_start, granularity_column)