from models.prophet_model import ProphetForecaster
from utils.split_utils import split_data
from utils.metrics_utils import evaluate_forecast
from utils.plot_utils import plot_forecast
import pandas as pd
pass

def _run_forecast(df, split_dates=('2015-06-30', '2015-12-31'), plot=True):
    import matplotlib.pyplot as plt
    train_end, val_end = split_dates
    train, val, test = split_data(df, train_end, val_end)
    full_truth = pd.concat([train, val, test])
    forecaster = ProphetForecaster()
    forecaster.fit(train)
    forecast = forecaster.forecast(periods=len(val) + len(test), known_dates=full_truth['ds'])
    forecast_full = forecaster.merge_forecast_with_truth(forecast, full_truth)
    scores = evaluate_forecast(forecast_full[forecast_full['ds'] > train['ds'].max()])
    fig = None
    if plot:
        fig = plot_forecast(full_truth, forecast_full, train['ds'].max())
    return (forecast_full, scores, fig)

def forecast_by_product(df, product_id, split_dates=('2015-06-30', '2015-12-31'), plot=True):
    product_df = df[df['id'] == product_id][['ds', 'sales']].rename(columns={'sales': 'y'})
    return _run_forecast(product_df, split_dates, plot)
pass

def forecast_by_category(df, category_name, split_dates=('2015-06-30', '2015-12-31'), plot=True):
    category_col = 'cat_id' if 'cat_id' in df.columns else 'product_category'
    category_df = df[df[category_col] == category_name].groupby('ds')['sales'].sum().reset_index()
    category_df = category_df.rename(columns={'sales': 'y'})
    return _run_forecast(category_df, split_dates, plot)

def forecast_by_store(df, store_id, split_dates=('2015-06-30', '2015-12-31'), plot=True):
    store_df = df[df['store_id'] == store_id].groupby('ds')['sales'].sum().reset_index()
    store_df = store_df.rename(columns={'sales': 'y'})
    return _run_forecast(store_df, split_dates, plot)

def run_prophet_forecast(df, mode='product', value=None, split_dates=('2015-06-30', '2015-12-31'), plot=True):
    """
    # Adaptation pour le dataset synthétique
    if 'cat_id' in df.columns:  # Dataset synthétique
        if mode == 'category':
            product_df = df[df['cat_id'] == value][['ds', 'y']]
        else:
            raise ValueError("Le dataset synthétique ne supporte que la granularité 'category'")
    else:  # Dataset M5
        if mode == 'product':
            product_df = df[df['id'] == value][['ds', 'sales']].rename(columns={'sales': 'y'})
    """
    if mode == 'product' and value:
        return forecast_by_product(df, value, split_dates, plot)
    if mode == 'category' and value:
        return forecast_by_category(df, value, split_dates, plot)
    if mode == 'store' and value:
        return forecast_by_store(df, value, split_dates, plot)
    raise ValueError('Invalid mode or missing value.')