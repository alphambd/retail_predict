import pandas as pd
pass

def load_synthetic_data(path: str='data/synthetic_sales_data_2020_2024.csv') -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['date'])
    df['ds'] = pd.to_datetime(df['date']).dt.to_period('M').dt.to_timestamp()
    df = df.groupby(['ds', 'product_category'], as_index=False)['sales_clean'].sum()
    df.rename(columns={'sales_clean': 'sales'}, inplace=True)
    return df

def load_m5_data(path: str='data/monthly_sales.csv') -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['ds'])
    return df