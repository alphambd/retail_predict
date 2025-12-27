import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from pathlib import Path
from gluonts.model.predictor import Predictor
from utils.metrics_utils import evaluate_forecast
from utils.plot_utils import plot_forecast

def preprocess_series(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds']).dt.to_period('M').dt.to_timestamp()
    df = df.sort_values('ds').reset_index(drop=True)
    return df[['ds', 'y']]

def split_data(df: pd.DataFrame, val_start: str, test_start: str):
    train = df[df['ds'] < val_start]
    val = df[(df['ds'] >= val_start) & (df['ds'] < test_start)]
    test = df[df['ds'] >= test_start]
    return (train, val, test)

def to_gluonts_dataset(df: pd.DataFrame, prediction_length: int, freq='M', item_id='series'):
    return ListDataset([{'start': df['ds'].iloc[0], 'target': df['y'].values, 'item_id': item_id}], freq=freq)

def load_or_train_model(train_dataset, context_length, prediction_length, model_name, freq='MS', epochs=100):
    model_path = Path(f'models/deepar_{model_name}')
    if model_path.exists():
        predictor = Predictor.deserialize(model_path)
        return predictor
    predictor = train_deepar_model(train_dataset, context_length, prediction_length, freq=freq, epochs=epochs, model_name=model_name)
    return predictor

def train_deepar_model(train_dataset, context_length: int, prediction_length: int, model_name: str, freq='MS', epochs=100):
    estimator = DeepAREstimator(prediction_length=prediction_length, context_length=context_length, freq=freq, dropout_rate=0.1, time_features=time_features_from_frequency_str(freq), num_layers=2, hidden_size=40, batch_size=32, num_batches_per_epoch=50, trainer_kwargs={'max_epochs': epochs})
    predictor = estimator.train(training_data=train_dataset)
    model_path = Path(f'models/deepar_{model_name}')
    model_path.mkdir(parents=True, exist_ok=True)
    predictor.serialize(model_path)
    return predictor

def prepare_forecast_df(full_df, forecast, prediction_length):
    forecast_dates = pd.date_range(start=full_df['ds'].iloc[-prediction_length], periods=prediction_length, freq='MS')
    forecast_df = pd.DataFrame({'ds': forecast_dates, 'yhat': forecast.mean, 'y': full_df['y'].values[-prediction_length:]})
    return forecast_df

def forecast_deepar(predictor, test_dataset, full_df, prediction_length):
    forecast_it, ts_it = make_evaluation_predictions(test_dataset, predictor=predictor, num_samples=100)
    forecasts = list(forecast_it)
    forecast_df = prepare_forecast_df(full_df, forecasts[0], prediction_length)
    return forecast_df

def forecast_by_product_deepar(df, product_id, val_start='2015-07-01', test_start='2016-01-01'):
    df_product = df[df['id'] == product_id][['ds', 'sales']].rename(columns={'sales': 'y'})
    df_product = preprocess_series(df_product)
    train_df, val_df, test_df = split_data(df_product, val_start, test_start)
    prediction_length = len(val_df) + len(test_df)
    context_length = 12
    full_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    train_ds = to_gluonts_dataset(train_df, prediction_length, item_id=product_id)
    test_ds = to_gluonts_dataset(full_df, prediction_length, item_id=product_id)
    predictor = load_or_train_model(train_ds, context_length, prediction_length, model_name=product_id)
    forecast_df = forecast_deepar(predictor, test_ds, full_df, prediction_length)
    scores = evaluate_forecast(forecast_df)
    fig = plot_forecast(data=full_df, forecast=forecast_df, train_end=train_df['ds'].max(), title='Prévision DeepAR', label=f'Produit {product_id}', model='DeepAR')
    return (forecast_df, scores, fig)

def forecast_by_category_deepar(df, category_name, val_start='2023-01-01', test_start='2024-01-01', category_column='category'):
    df_cat = df[df[category_column] == category_name].groupby('ds', as_index=False)['sales'].sum().rename(columns={'sales': 'y'})
    df_cat = preprocess_series(df_cat)
    train_df, val_df, test_df = split_data(df_cat, val_start, test_start)
    prediction_length = len(val_df) + len(test_df)
    context_length = 12
    full_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    train_ds = to_gluonts_dataset(train_df, prediction_length, item_id=category_name)
    test_ds = to_gluonts_dataset(full_df, prediction_length, item_id=category_name)
    predictor = load_or_train_model(train_ds, context_length, prediction_length, model_name=f'synthetic_{category_name}')
    forecast_df = forecast_deepar(predictor, test_ds, full_df, prediction_length)
    scores = evaluate_forecast(forecast_df)
    fig = plot_forecast(data=full_df, forecast=forecast_df, train_end=train_df['ds'].max(), title='Prévision DeepAR', label=f'Catégorie {category_name}', model='DeepAR')
    return (forecast_df, scores, fig)

def forecast_by_store_deepar(df, store_id, val_start='2015-07-01', test_start='2016-01-01'):
    df_store = df[df['store_id'] == store_id].groupby('ds')['sales'].sum().reset_index().rename(columns={'sales': 'y'})
    df_store = preprocess_series(df_store)
    train_df, val_df, test_df = split_data(df_store, val_start, test_start)
    prediction_length = len(val_df) + len(test_df)
    context_length = 12
    full_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    train_ds = to_gluonts_dataset(train_df, prediction_length, item_id=store_id)
    test_ds = to_gluonts_dataset(full_df, prediction_length, item_id=store_id)
    predictor = load_or_train_model(train_ds, context_length, prediction_length, model_name=store_id)
    forecast_df = forecast_deepar(predictor, test_ds, full_df, prediction_length)
    scores = evaluate_forecast(forecast_df)
    fig = plot_forecast(data=full_df, forecast=forecast_df, train_end=train_df['ds'].max(), title='Prévision DeepAR', label=f'Magasin {store_id}', model='DeepAR')
    return (forecast_df, scores, fig)
pass

def run_deepar_forecast(df, granularity='product', value=None, val_start='2015-07-01', test_start='2016-01-01'):
    is_synthetic = 'category' in df.columns and 'cat_id' not in df.columns
    if is_synthetic and granularity == 'category' and value:
        return forecast_by_category_deepar(df, value, val_start='2023-01-01', test_start='2024-01-01')
    if not is_synthetic:
        if granularity == 'product' and value:
            return forecast_by_product_deepar(df, value, val_start, test_start)
        if granularity == 'category' and value:
            granularity_column = 'product_category' if 'product_category' in df.columns else 'cat_id'
            return forecast_by_category_deepar(df, value, val_start, test_start, category_column=granularity_column)
        if granularity == 'store' and value:
            return forecast_by_store_deepar(df, value, val_start, test_start)
    raise ValueError('Granularité invalide ou identifiant manquant.')