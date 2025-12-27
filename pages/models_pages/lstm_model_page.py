import streamlit as st
import pandas as pd
from services.lstm_forecasting_service import run_lstm_forecast
from services.data_loader import load_synthetic_data, load_m5_data

def show():
    st.title('🧠 Modèle LSTM Bidirectionnel')
    st.markdown('\n    Ce modèle utilise un réseau de neurones récurrent bidirectionnel pour capturer les motifs temporels dans les deux directions (passé → futur et futur → passé).\n\n    **Paramètres par défaut :**\n    - Fenêtre temporelle : 12 mois\n    - Couches : 2 couches Bidirectionnelles LSTM (64 et 32 unités)\n    - Dropout : 20%\n    - Epochs : 100\n    ')
    dataset_option = st.selectbox('Sélectionnez le jeu de données :', ['M5 dataset', 'Synthetic dataset'])
    pass
    if dataset_option == 'Synthetic dataset':
        df = load_synthetic_data()
        granularities = ['category']
        granularity_column_map = {'category': 'product_category'}
    else:
        df = load_m5_data()
        granularities = ['product', 'category', 'store']
        granularity_column_map = {'product': 'id', 'category': 'cat_id', 'store': 'store_id'}
    val_start = '2015-07-01' if dataset_option == 'M5 dataset' else '2023-01-01'
    test_start = '2016-01-01' if dataset_option == 'M5 dataset' else '2024-01-01'
    granularity = st.selectbox('Niveau de prévision :', granularities)
    granularity_column = granularity_column_map[granularity]
    available_ids = df[granularity_column].unique().tolist()
    target_id = st.selectbox(f'Sélectionnez une {granularity} :', available_ids)
    with st.expander('Paramètres avancés'):
        epochs = st.slider("Nombre d'epochs", 50, 500, 100)
        window_size = st.slider('Taille de la fenêtre temporelle', 6, 24, 12)
        batch_size = st.selectbox('Taille du batch', [4, 8, 16, 32], index=1)
    if st.button('Lancer la prévision avec LSTM Bidirectionnel'):
        with st.spinner('Entraînement du modèle en cours (cela peut prendre quelques minutes)...'):
            try:
                forecast_df, metrics, fig = run_lstm_forecast(df, granularity, target_id, val_start, test_start)
                st.success('Modèle exécuté avec succès ✅')
                st.subheader('📊 Résultats de la prédiction')
                st.pyplot(fig)
                st.subheader('📈 Métriques')
                if isinstance(metrics, dict):
                    metrics = pd.DataFrame(list(metrics.items()), columns=['Métrique', 'Valeur'])
                st.dataframe(metrics)
            except Exception as e:
                st.error(f'Une erreur est survenue : {str(e)}')