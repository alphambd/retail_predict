import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from utils.metrics_utils import evaluate_forecast
from utils.plot_utils import plot_forecast

class ExtremeLearningMachine:

    def __init__(self, n_hidden=100, activation='relu', alpha=1.0, random_state=42):
        self.n_hidden = n_hidden
        self.activation = activation
        self.alpha = alpha
        self.random_state = random_state
        self.input_weights = None
        self.biases = None
        self.output_weights = None
        self.scaler = StandardScaler()

    def _activation_func(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        if self.activation == 'tanh':
            return np.tanh(x)
        raise ValueError(f'Unsupported activation function: {self.activation}')

    def fit(self, X, y):
        np.random.seed(self.random_state)
        X_scaled = self.scaler.fit_transform(X)
        n_features = X_scaled.shape[1]
        self.input_weights = np.random.normal(size=(n_features, self.n_hidden))
        self.biases = np.random.normal(size=(self.n_hidden,))
        H = self._activation_func(np.dot(X_scaled, self.input_weights) + self.biases)
        ridge = Ridge(alpha=self.alpha, fit_intercept=False)
        ridge.fit(H, y)
        self.output_weights = ridge.coef_

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        H = self._activation_func(np.dot(X_scaled, self.input_weights) + self.biases)
        return np.dot(H, self.output_weights)

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

def train_elm_forecast(train_df, val_df, test_df):
    features = [col for col in train_df.columns if col not in ['ds', 'y']]
    X_train, y_train = (train_df[features], train_df['y'])
    X_val, y_val = (val_df[features], val_df['y'])
    X_test, y_test = (test_df[features], test_df['y'])
    model = ExtremeLearningMachine(n_hidden=100, activation='relu', alpha=1.0)
    model.fit(X_train, y_train)
    yhat_val = model.predict(X_val)
    yhat_test = model.predict(X_test)
    forecast_df = pd.concat([val_df[['ds', 'y']].assign(yhat=yhat_val), test_df[['ds', 'y']].assign(yhat=yhat_test)])
    return (forecast_df, model)

def forecast_generic(df, filter_col, filter_val, val_start, test_start):
    filtered_df = df[df[filter_col] == filter_val]
    if filter_col == 'id':
        grouped_df = filtered_df[['ds', 'sales']].rename(columns={'sales': 'y'})
        label = f'Produit {filter_val}'
    else:
        grouped_df = filtered_df.groupby('ds', as_index=False)['sales'].sum().rename(columns={'sales': 'y'})
        label = f'{filter_col.capitalize()} {filter_val}'
    lagged_df = create_lag_features(grouped_df)
    train, val, test = split_data(lagged_df, val_start, test_start)
    full_truth = pd.concat([train, val, test])
    forecast_df, model = train_elm_forecast(train, val, test)
    scores = evaluate_forecast(forecast_df)
    fig = plot_forecast(data=full_truth, forecast=forecast_df, train_end=train['ds'].max(), title='Prévision', label=label, model='ELM')
    return (forecast_df, scores, fig)
pass

def run_elm_forecast(df, granularity='product', value=None, val_start='2015-07-01', test_start='2016-01-01'):
    if granularity == 'product':
        granularity_column = 'id' if 'id' in df.columns else 'product_id'
    elif granularity == 'category':
        granularity_column = 'cat_id' if 'cat_id' in df.columns else 'product_category'
    elif granularity == 'store':
        granularity_column = 'store_id' if 'store_id' in df.columns else 'store_name'
    else:
        raise ValueError('Granularité invalide.')
    if value is not None:
        return forecast_generic(df, granularity_column, value, val_start, test_start)
    raise ValueError('Valeur de granularité manquante.')