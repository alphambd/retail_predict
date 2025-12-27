from prophet import Prophet
import pandas as pd

class ProphetForecaster:

    def __init__(self, holidays=None, **kwargs):
        self.model = Prophet(holidays=holidays, **kwargs)
        self.fitted = False

    def fit(self, train_df):
        df = train_df[['ds', 'y']].copy()
        self.model.fit(df)
        self.fitted = True

    def forecast(self, periods, freq='MS', known_dates=None):
        if not self.fitted:
            raise RuntimeError('Le modèle doit être entraîné avant la prédiction.')
        if known_dates is not None:
            future = pd.DataFrame({'ds': known_dates})
        else:
            future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        return forecast

    def merge_forecast_with_truth(self, forecast, true_df):
        return forecast.set_index('ds').join(true_df.set_index('ds')).reset_index()