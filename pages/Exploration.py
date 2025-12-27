
import os
import pandas as pd
import streamlit as st
import plotly.express as px
from utils.data_loader import load_sales_data


# Titre de la page
st.title("🔍 Exploration des ventes")

# --- 1. Choix du dataset ---
st.sidebar.header("🔢 Sélection du dataset")
dataset_option = st.sidebar.selectbox("Choisissez un dataset :", ["M5 Forecasting", "Synthétique"])

# --- 2. Chargement des données en fonction du choix ---
if dataset_option == "M5 Forecasting":
    df = load_sales_data(dataset_choice="M5")
    df["store"] = df.get("store", "M5")  # pour éviter KeyError si absent
    df["source"] = "M5"
else:
    df = load_sales_data(dataset_choice="Synthétique")
    #df["store"] = "Synthétique"  # pas de notion de magasin
    #df["source"] = "Synthétique"

# --- 3. Aperçu des données ---
if st.checkbox("Afficher un aperçu des données"):
    st.dataframe(df.head())

# Harmoniser colonnes utiles
df = df[["date", "category", "store", "sales"]]



# --- 4. Filtres ---
st.sidebar.header("🧰 Filtres")

categories = df["category"].unique()
selected_category = st.sidebar.selectbox("Catégorie", sorted(categories))

#stores = df["store"].unique()
#selected_store = st.sidebar.selectbox("Magasin", sorted(stores))
if dataset_option == "M5 Forecasting":
    stores = df["store"].unique()
    selected_store = st.sidebar.selectbox("Magasin", sorted(stores))
else:
    selected_store = "Synthétique"


# --- 5. Filtrage ---
filtered_df = df[
    (df["category"] == selected_category) &
    (df["store"] == selected_store)
]

# --- 6. Plage temporelle ---
min_date, max_date = filtered_df["date"].min().date(), filtered_df["date"].max().date()

date_range = st.sidebar.slider(
    "Plage temporelle",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM"
)

start_date = pd.Timestamp(date_range[0])
end_date = pd.Timestamp(date_range[1])

filtered_df = filtered_df[
    (filtered_df["date"] >= start_date) & (filtered_df["date"] <= end_date)
]

# --- 7. Agrégation et visualisation ---
monthly_sales = (
    filtered_df.groupby("date")["sales"]
    .sum()
    .reset_index()
    .sort_values("date")
)

st.subheader(f"📈 Ventes mensuelles – {selected_category} | {selected_store}")
fig = px.line(
    monthly_sales,
    x="date",
    y="sales",
    labels={"date": "Date", "sales": "Ventes"},
    title="Évolution des ventes mensuelles"
)
st.plotly_chart(fig, use_container_width=True)

# --- 8. Statistiques ---
st.subheader("📊 Statistiques descriptives")
st.write(filtered_df["sales"].describe())


