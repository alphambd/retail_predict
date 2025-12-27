from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

def evaluate_forecast(forecast_df, y_col='y', yhat_col='yhat'):
    y_true = forecast_df[y_col]
    y_pred = forecast_df[yhat_col]
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {'mse': mse, 'rmse': rmse, 'mape': mape}