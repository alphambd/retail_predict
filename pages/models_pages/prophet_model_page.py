import streamlit as st
import pandas as pd
from services.prophet_forecasting_service import run_prophet_forecast
from services.data_loader import load_synthetic_data, load_m5_data
from utils.evaluation import compute_metrics
import matplotlib.pyplot as plt

def show():
    dataset_option = st.selectbox('Sélectionnez le jeu de données :', ['M5 dataset', 'Synthetic dataset'])
    pass
    if dataset_option == 'Synthetic dataset':
        df = load_synthetic_data()
        split_dates = ('2023-01-01', '2024-01-01')
        granularities = ['category']
        granularity_column_map = {'category': 'product_category'}
    else:
        df = load_m5_data()
        split_dates = ('2015-06-30', '2015-12-31')
        granularities = ['product', 'category', 'store']
        granularity_column_map = {'product': 'id', 'category': 'cat_id', 'store': 'store_id'}
    granularity = st.selectbox('Niveau de prévision :', granularities)
    granularity_column = granularity_column_map[granularity]
    available_ids = df[granularity_column].unique().tolist()
    target_id = st.selectbox(f'Sélectionnez une {granularity} :', available_ids)
    if st.button('Lancer la prévision avec Prophet'):
        with st.spinner('Entraînement et prédiction en cours...'):
            forecast_df, metrics, fig = run_prophet_forecast(df, granularity, target_id, split_dates=split_dates)
        st.success('Modèle exécuté avec succès ✅')
        st.subheader('📊 Résultats de la prédiction')
        st.pyplot(fig)
        st.subheader('📈 Métriques')
        if isinstance(metrics, dict):
            metrics = pd.DataFrame(list(metrics.items()), columns=['Métrique', 'Valeur'])
        st.dataframe(metrics)