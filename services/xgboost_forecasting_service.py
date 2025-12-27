import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from utils.metrics_utils import evaluate_forecast
from utils.plot_utils import plot_forecast

def create_lag_features(df, lags=6):
    df = df.copy()
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['y'].shift(i)
    df['month'] = pd.to_datetime(df['ds']).dt.month
    return df.dropna().reset_index(drop=True)

def split_xgboost_data(df, val_start, test_start):
    train = df[df['ds'] < val_start]
    val = df[(df['ds'] >= val_start) & (df['ds'] < test_start)]
    test = df[df['ds'] >= test_start]
    return (train, val, test)

def train_xgboost_forecast(train_df, val_df, test_df, target_col='y'):
    features = [col for col in train_df.columns if col not in ['ds', 'y']]
    X_train, y_train = (train_df[features], train_df[target_col])
    X_val, y_val = (val_df[features], val_df[target_col])
    X_test, y_test = (test_df[features], test_df[target_col])
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    yhat_val = model.predict(X_val)
    yhat_test = model.predict(X_test)
    forecast_df = pd.concat([val_df[['ds', 'y']].assign(yhat=yhat_val), test_df[['ds', 'y']].assign(yhat=yhat_test)])
    return (forecast_df, model)

def forecast_by_product_xgb(df, product_id, val_start, test_start):
    product_df = df[df['id'] == product_id][['ds', 'sales']].rename(columns={'sales': 'y'})
    product_df_lagged = create_lag_features(product_df)
    train, val, test = split_xgboost_data(product_df_lagged, val_start, test_start)
    full_truth = pd.concat([train, val, test])
    forecast_df, model = train_xgboost_forecast(train, val, test)
    scores = evaluate_forecast(forecast_df)
    fig = plot_forecast(data=full_truth, forecast=forecast_df, train_end=train['ds'].max(), title='Prévision', label=f'Produit {product_id}', model='XGBoost')
    return (forecast_df, scores, fig)

def forecast_by_category_xgb(df, category_id, val_start, test_start, category_column='cat_id'):
    category_df = df[df[category_column] == category_id].groupby('ds', as_index=False)['sales'].sum().rename(columns={'sales': 'y'})
    category_df_lagged = create_lag_features(category_df)
    train, val, test = split_xgboost_data(category_df_lagged, val_start, test_start)
    full_truth = pd.concat([train, val, test])
    forecast_df, model = train_xgboost_forecast(train, val, test)
    scores = evaluate_forecast(forecast_df)
    fig = plot_forecast(data=full_truth, forecast=forecast_df, train_end=train['ds'].max(), title='Prévision', label=f'Catégorie {category_id}', model='XGBoost')
    return (forecast_df, scores, fig)

def forecast_by_store_xgb(df, store_id, val_start, test_start):
    store_df = df[df['store_id'] == store_id].groupby('ds', as_index=False)['sales'].sum().rename(columns={'sales': 'y'})
    store_df_lagged = create_lag_features(store_df)
    train, val, test = split_xgboost_data(store_df_lagged, val_start, test_start)
    full_truth = pd.concat([train, val, test])
    forecast_df, model = train_xgboost_forecast(train, val, test)
    scores = evaluate_forecast(forecast_df)
    fig = plot_forecast(data=full_truth, forecast=forecast_df, train_end=train['ds'].max(), title='Prévision', label=f'Magasin {store_id}', model='XGBoost')
    return (forecast_df, scores, fig)

def run_xgboost_forecast(df, granularity='product', value=None, val_start='2015-07-01', test_start='2016-01-01'):
    if granularity == 'product' and value:
        return forecast_by_product_xgb(df, value, val_start, test_start)
    if granularity == 'category' and value:
        granularity_column = 'product_category' if 'product_category' in df.columns else 'cat_id'
        return forecast_by_category_xgb(df, value, val_start, test_start, granularity_column)
    if granularity == 'store' and value:
        return forecast_by_store_xgb(df, value, val_start, test_start)
    raise ValueError('Granularité invalide ou identifiant manquant.')