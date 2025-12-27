from models.prophet_model import ProphetForecaster
from utils.split_utils import split_data
from utils.metrics_utils import evaluate_forecast
from utils.plot_utils import plot_forecast
import pandas as pd

def _run_forecast(df, split_dates=('2015-06-30', '2015-12-31'), plot=True):
    train_end, val_end = split_dates
    train, val, test = split_data(df, train_end, val_end)
    full_truth = pd.concat([train, val, test])
    forecaster = ProphetForecaster()
    forecaster.fit(train)
    forecast = forecaster.forecast(periods=len(val) + len(test), known_dates=full_truth['ds'])
    forecast_full = forecaster.merge_forecast_with_truth(forecast, full_truth)
    scores = evaluate_forecast(forecast_full[forecast_full['ds'] > train['ds'].max()])
    if plot:
        plot_forecast(full_truth, forecast_full, train['ds'].max())
    return scores

def forecast_by_product(df, product_id, split_dates=('2015-06-30', '2015-12-31'), plot=True):
    product_df = df[df['id'] == product_id][['ds', 'sales']].rename(columns={'sales': 'y'})
    return _run_forecast(product_df, split_dates, plot)

def forecast_by_category(df, category_name, split_dates=('2015-06-30', '2015-12-31'), plot=True):
    category_df = df[df['cat_id'] == category_name].groupby('ds')['sales'].sum().reset_index()
    category_df = category_df.rename(columns={'sales': 'y'})
    return _run_forecast(category_df, split_dates, plot)

def forecast_by_store(df, store_id, split_dates=('2015-06-30', '2015-12-31'), plot=True):
    store_df = df[df['store_id'] == store_id].groupby('ds')['sales'].sum().reset_index()
    store_df = store_df.rename(columns={'sales': 'y'})
    return _run_forecast(store_df, split_dates, plot)