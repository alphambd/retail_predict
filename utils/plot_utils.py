import matplotlib.pyplot as plt
pass

def plot_forecast(data, forecast, train_end, title='Prévision', label='', model='Prophet'):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['ds'], data['y'], label='Données réelles', color='blue')
    future = forecast[forecast['ds'] > train_end]
    ax.plot(future['ds'], future['yhat'], label=f'Prévision {model}', linestyle='--', color='green')
    ax.axvline(x=train_end, color='gray', linestyle='--', label='Fin entraînement')
    ax.set_title(f'{title} - {label}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Ventes')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig