import pandas as pd
import os
pass
import os
import pandas as pd
import streamlit as st

def load_sales_data(dataset_choice):
    if dataset_choice == 'M5':
        file_path = os.path.join('data', 'monthly_sales.csv')
    elif dataset_choice == 'Synthétique':
        file_path = os.path.join('data', 'synthetic_sales_data_2020_2024.csv')
    else:
        st.error('Choix de dataset non reconnu.')
        st.stop()
    try:
        df = pd.read_csv(file_path)
        if 'ds' in df.columns and 'date' not in df.columns:
            df.rename(columns={'ds': 'date'}, inplace=True)
        df = df.loc[:, ~df.columns.duplicated()]
        if 'product_category' in df.columns:
            df.rename(columns={'product_category': 'category'}, inplace=True)
        if 'sales_clean' in df.columns:
            df.rename(columns={'sales_clean': 'sales'}, inplace=True)
        if 'cat_id' in df.columns:
            df.rename(columns={'cat_id': 'category'}, inplace=True)
        if 'store_id' in df.columns:
            df.rename(columns={'store_id': 'store'}, inplace=True)
        if 'ds' in df.columns and 'date' not in df.columns:
            df.rename(columns={'ds': 'date'}, inplace=True)
        if 'date' in df.columns and (not pd.api.types.is_datetime64_any_dtype(df['date'])):
            df['date'] = pd.to_datetime(df['date'])
        elif 'date' not in df.columns:
            st.error(f"Aucune colonne 'date' trouvée dans le fichier {file_path}. Colonnes disponibles : {df.columns}")
            st.stop()
        if dataset_choice == 'Synthétique':
            df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
            df = df.groupby(['category', 'month'])['sales'].sum().reset_index()
            df.rename(columns={'month': 'date'}, inplace=True)
            df['store'] = 'Synthétique'
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        if df.columns.tolist().count('sales') > 1:
            first_sales_col = df.loc[:, 'sales'].iloc[:, 0]
            df = df.drop(columns=['sales'])
            df['sales'] = first_sales_col
        return df
    except Exception as e:
        st.error(f'Erreur lors du chargement des données : {e}')
        st.stop()