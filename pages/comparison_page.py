import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from services.data_loader import load_synthetic_data, load_m5_data
from services.prophet_forecasting_service import run_prophet_forecast
from services.xgboost_forecasting_service import run_xgboost_forecast
from services.deepar_forecasting_service import run_deepar_forecast
from services.lstm_forecasting_service import run_lstm_forecast
from services.fnn_forecasting_service import run_fnn_forecast
from services.elm_forecasting_service import run_elm_forecast
from services.svm_forecasting_service import run_svm_forecast

def show_comparison():
    """
    Affiche la page de comparaison des modèles de prévision.
    Permet de comparer plusieurs modèles sur un même graphique.
    """
    
    st.title('📊 Comparaison des Modèles')
    st.markdown('Comparez les performances des différents modèles sur un même graphique')
    
    # 1. Sélection du dataset
    dataset_option = st.selectbox(
        'Sélectionnez le jeu de données :', 
        ['M5 dataset', 'Synthetic dataset']
    )
    
    # Charger les données
    if dataset_option == 'Synthetic dataset':
        df = load_synthetic_data()
        granularities = ['category']
        granularity_column_map = {'category': 'product_category'}
    else:
        df = load_m5_data()
        granularities = ['product', 'category', 'store']
        granularity_column_map = {'product': 'id', 'category': 'cat_id', 'store': 'store_id'}
    
    # 2. Sélection de la granularité
    granularity = st.selectbox('Niveau de prévision :', granularities)
    granularity_column = granularity_column_map[granularity]
    
    # 3. Sélection de l'ID cible
    available_ids = df[granularity_column].unique().tolist()
    if not available_ids:
        st.warning(f"Aucun {granularity} disponible dans les données")
        return
    
    target_id = st.selectbox(f'Sélectionnez une {granularity} :', available_ids)
    
    # Filtrer les données
    df_filtered = df[df[granularity_column] == target_id]
    
    # 4. Préparer les données de base pour le graphique
    if dataset_option == 'Synthetic dataset':
        base_df = df_filtered.groupby('ds')['sales'].sum().reset_index()
        base_df = base_df.rename(columns={'sales': 'y'})
    elif granularity == 'product':
        base_df = df_filtered[['ds', 'sales']].rename(columns={'sales': 'y'})
    else:
        base_df = df_filtered.groupby('ds')['sales'].sum().reset_index().rename(columns={'sales': 'y'})
    
    # 5. Sélection des modèles à comparer
    models_to_compare = st.multiselect(
        'Sélectionnez les modèles à comparer :', 
        ['Prophet', 'XGBoost', 'DeepAR', 'LSTM Bidirectionnel', 'FNN', 'ELM', 'SVM'], 
        default=['Prophet', 'XGBoost']
    )
    
    # 6. Définir les dates de split
    val_start = '2015-07-01' if dataset_option == 'M5 dataset' else '2023-01-01'
    test_start = '2016-01-01' if dataset_option == 'M5 dataset' else '2024-01-01'
    split_dates = (val_start, test_start)
    
    # 7. Bouton pour lancer la comparaison
    if st.button('🚀 Lancer la comparaison', type='primary'):
        if not models_to_compare:
            st.warning("Veuillez sélectionner au moins un modèle")
            return
        
        results = {}
        metrics = {}
        
        with st.spinner('Calcul des prévisions...'):
            # Exécuter chaque modèle sélectionné
            if 'Prophet' in models_to_compare:
                try:
                    prophet_df, prophet_metrics, _ = run_prophet_forecast(
                        df, granularity, target_id, split_dates=split_dates
                    )
                    if prophet_df is not None:
                        prophet_df = prophet_df[prophet_df['ds'] >= pd.to_datetime(val_start)]
                        results['Prophet'] = prophet_df
                        metrics['Prophet'] = prophet_metrics
                except Exception as e:
                    st.error(f"Erreur Prophet: {e}")
            
            if 'XGBoost' in models_to_compare:
                try:
                    xgb_df, xgb_metrics, _ = run_xgboost_forecast(
                        df, granularity, target_id, val_start, test_start
                    )
                    if xgb_df is not None:
                        results['XGBoost'] = xgb_df
                        metrics['XGBoost'] = xgb_metrics
                except Exception as e:
                    st.error(f"Erreur XGBoost: {e}")
            
            if 'DeepAR' in models_to_compare:
                try:
                    deepar_df, deepar_metrics, _ = run_deepar_forecast(
                        df, granularity, target_id, val_start, test_start
                    )
                    if deepar_df is not None:
                        results['DeepAR'] = deepar_df
                        metrics['DeepAR'] = deepar_metrics
                except Exception as e:
                    st.error(f"Erreur DeepAR: {e}")
            
            if 'LSTM Bidirectionnel' in models_to_compare:
                try:
                    lstm_df, lstm_metrics, _ = run_lstm_forecast(
                        df, granularity, target_id, val_start, test_start
                    )
                    if lstm_df is not None:
                        results['LSTM'] = lstm_df
                        metrics['LSTM'] = lstm_metrics
                except Exception as e:
                    st.error(f"Erreur LSTM: {e}")
            
            if 'FNN' in models_to_compare:
                try:
                    fnn_df, fnn_metrics, _ = run_fnn_forecast(
                        df, granularity, target_id, val_start, test_start
                    )
                    if fnn_df is not None:
                        results['FNN'] = fnn_df
                        metrics['FNN'] = fnn_metrics
                except Exception as e:
                    st.error(f"Erreur FNN: {e}")
            
            if 'ELM' in models_to_compare:
                try:
                    elm_df, elm_metrics, _ = run_elm_forecast(
                        df, granularity, target_id, val_start, test_start
                    )
                    if elm_df is not None:
                        results['ELM'] = elm_df
                        metrics['ELM'] = elm_metrics
                except Exception as e:
                    st.error(f"Erreur ELM: {e}")
            
            if 'SVM' in models_to_compare:
                try:
                    svm_df, svm_metrics, _ = run_svm_forecast(
                        df, granularity, target_id, val_start, test_start
                    )
                    if svm_df is not None:
                        results['SVM'] = svm_df
                        metrics['SVM'] = svm_metrics
                except Exception as e:
                    st.error(f"Erreur SVM: {e}")
        
        # Vérifier si on a des résultats
        if not results:
            st.warning("Aucun modèle n'a pu être exécuté. Vérifiez les logs pour plus d'informations.")
            return
        
        # 8. Créer le graphique de comparaison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Couleurs pour chaque modèle
        colors = {
            'Prophet': 'blue', 
            'XGBoost': 'green', 
            'DeepAR': 'red', 
            'LSTM': 'purple', 
            'FNN': 'orange', 
            'ELM': 'brown', 
            'SVM': 'pink'
        }
        
        # Tracer les données réelles
        ax.plot(base_df['ds'], base_df['y'], 
                label='Données réelles', color='black', linewidth=2)
        
        # Tracer les prédictions de chaque modèle
        for model_name, df_result in results.items():
            if model_name in colors and not df_result.empty:
                # Vérifier les noms de colonnes (yhat ou sales_pred, etc.)
                pred_column = 'yhat' if 'yhat' in df_result.columns else \
                             'sales_pred' if 'sales_pred' in df_result.columns else \
                             'prediction' if 'prediction' in df_result.columns else \
                             df_result.columns[1]  # Deuxième colonne par défaut
                
                ax.plot(df_result['ds'], df_result[pred_column], 
                       label=model_name, linestyle='--', color=colors[model_name])
        
        # Ajouter une ligne verticale pour la fin de l'entraînement
        train_end = base_df[base_df['ds'] < pd.to_datetime(val_start)]['ds'].max()
        ax.axvline(x=train_end, color='gray', linestyle='--', 
                  label='Fin entraînement', alpha=0.7)
        
        # Configurer le graphique
        ax.set_title(f'Comparaison des modèles - {granularity} {target_id}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Ventes')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Afficher le graphique
        st.pyplot(fig)
        
        # 9. Afficher les métriques comparatives
        if metrics:
            st.subheader('📈 Métriques comparatives')
            
            try:
                # Créer un DataFrame avec les métriques
                metrics_df = pd.DataFrame(metrics).T
                
                # Debug: Afficher les colonnes disponibles
                st.write(f"Colonnes disponibles dans metrics_df: {list(metrics_df.columns)}")
                
                # Normaliser les noms de colonnes (au cas où)
                column_mapping = {
                    'mae': 'MAE', 'rmse': 'RMSE', 'mape': 'MAPE', 'r2': 'R2',
                    'mean_absolute_error': 'MAE', 'root_mean_squared_error': 'RMSE',
                    'mean_absolute_percentage_error': 'MAPE', 'r_squared': 'R2'
                }
                
                # Renommer les colonnes si nécessaire
                metrics_df = metrics_df.rename(columns={k.lower(): v for k, v in column_mapping.items() 
                                                       if k.lower() in metrics_df.columns})
                metrics_df = metrics_df.rename(columns=column_mapping)
                
                # Formater les métriques
                numeric_cols = metrics_df.select_dtypes(include=['float64', 'int64']).columns
                
                if not numeric_cols.empty:
                    # Créer un style conditionnel
                    styled_df = metrics_df.style
                    
                    # Appliquer le highlight seulement si les colonnes existent
                    highlight_cols = {}
                    
                    if 'MAE' in metrics_df.columns:
                        styled_df = styled_df.highlight_min(subset=['MAE'], color='lightgreen')
                        highlight_cols['MAE'] = metrics_df['MAE'].idxmin()
                    
                    if 'RMSE' in metrics_df.columns:
                        styled_df = styled_df.highlight_min(subset=['RMSE'], color='lightgreen')
                        highlight_cols['RMSE'] = metrics_df['RMSE'].idxmin()
                    
                    if 'MAPE' in metrics_df.columns:
                        styled_df = styled_df.highlight_min(subset=['MAPE'], color='lightgreen')
                        highlight_cols['MAPE'] = metrics_df['MAPE'].idxmin()
                    
                    if 'R2' in metrics_df.columns:
                        styled_df = styled_df.highlight_max(subset=['R2'], color='lightgreen')
                        highlight_cols['R2'] = metrics_df['R2'].idxmax()
                    
                    # Formater les nombres
                    for col in numeric_cols:
                        if col in ['MAE', 'RMSE', 'MAPE']:
                            styled_df = styled_df.format({col: "{:.2f}"})
                        elif col == 'R2':
                            styled_df = styled_df.format({col: "{:.3f}"})
                    
                    # Afficher le tableau
                    st.dataframe(styled_df)
                    
                    # Ajouter une analyse simple
                    st.subheader('🏆 Analyse des résultats')
                    
                    if highlight_cols:
                        for metric, best_model in highlight_cols.items():
                            if best_model in metrics_df.index:
                                value = metrics_df.loc[best_model, metric]
                                if metric in ['MAE', 'RMSE', 'MAPE']:
                                    st.write(f"**Meilleur {metric} ({value:.2f}) :** {best_model}")
                                elif metric == 'R2':
                                    st.write(f"**Meilleur {metric} ({value:.3f}) :** {best_model}")
                    
                                            
                else:
                    # Si pas de colonnes numériques, afficher le DataFrame brut
                    st.dataframe(metrics_df)
                    
            except Exception as e:
                st.error(f"Erreur lors de l'affichage des métriques: {e}")
                # Afficher les métriques brutes pour debug
                st.write("**Métriques brutes :**")
                for model_name, model_metrics in metrics.items():
                    st.write(f"**{model_name}**: {model_metrics}")
        
        else:
            st.warning("Aucune métrique disponible pour l'affichage")
    
    else:
        # Message d'attente
        st.info("""
        👈 **Instructions :**
        1. Sélectionnez un jeu de données
        2. Choisissez le niveau de granularité
        3. Sélectionnez l'élément à analyser
        4. Choisissez les modèles à comparer
        5. Cliquez sur **"Lancer la comparaison"**
        """)

# Point d'entrée pour les tests
if __name__ == "__main__":
    show_comparison()