import streamlit as st

st.set_page_config(page_title="Accueil", page_icon="🏠", layout="wide")

st.title("🏠 Bienvenue sur RetailPredict")
st.markdown("---")

# Tabs principaux
tab1, tab2, tab3 = st.tabs(["📌 Présentation", "🧠 Modèles étudiés", "🧪 Expérimentations"])

# Onglet 1 : Présentation générale de l'application
with tab1:
    st.subheader("🎯 Objectifs de l'application")
    st.markdown("""
   **RetailPredict** est une application interactive dédiée à la **prévision des ventes** à l'aide de modèles de séries temporelles.  
    Elle a été développée dans le cadre d’un **projet académique (TER)** du **Master Données et Systèmes Connectés**.

    **Objectif principal** :  
    Offrir un outil permettant de :
    - simuler des prévisions,
    - comparer plusieurs modèles (classiques, ML, profonds),
    - visualiser les erreurs et résultats par granularité (**produit**, **catégorie**, **magasin**).
    """)

    st.subheader("🛠️ Fonctionnalités principales")
    st.markdown("""
    - 🔍 Exploration des données historiques
    - 📈 Application de différents modèles de prévision :
        - Prophet, XGBoost, LSTM, DeepAR, SVM...
    - 📊 Visualisation des performances (RMSE, MAPE, etc.)
    - 🧪 Comparaison des modèles sur plusieurs jeux de données
    - 📁 Gestion multi-niveaux : produit / catégorie / magasin
    """)

    st.subheader("📚 Contexte académique")
    st.info("""
    Projet TER réalisé dans le cadre du **Master 1 Données et Systèmes Connectés** – Université Jean Monnet  
    Année universitaire : 2024–2025  
    """)

# Onglet 2 : Modèles étudiés
with tab2:
    st.subheader("📈 Modèles implémentés")

    st.markdown("#### 🔹 Prophet")
    st.markdown("""
    - Modèle additif développé par Facebook.
    - Capture les tendances, saisonnalités (jour, semaine, année).
    - Robuste aux jours fériés et événements irréguliers.
    """)

    st.markdown("#### 🔹 XGBoost")
    st.markdown("""
    - Algorithme de gradient boosting basé sur les arbres de décision.
    - Utilisé avec des features créées (lags, encodage temporel...).
    - Bon compromis entre performance et temps d’exécution.
    """)

    st.markdown("#### 🔹 LSTM (Long Short-Term Memory)")
    st.markdown("""
    - Réseau de neurones récurrent adapté aux séquences.
    - Capte les dépendances temporelles longues.
    - Sensible au prétraitement (normalisation, taille fenêtre...).
    """)

    st.markdown("#### 🔹 DeepAR")
    st.markdown("""
    - Modèle probabiliste séquentiel basé sur des RNN (Amazon GluonTS).
    - Produit une **distribution de prévision**, pas juste une valeur unique.
    - Bien adapté à la prévision multi-séries avec incertitude.
    """)

    st.markdown("#### 🔹 SVM / ELM")
    st.markdown("""
    - Modèles linéaires ou à noyaux appliqués à des features temporelles.
    - Bon pour la comparaison de base, moins pour des séries complexes.
    - GridSearchCV avec cache pour optimiser les hyperparamètres.
    """)

# Onglet 3 : Expérimentations et résultats
with tab3:
    st.subheader("📊 Résultats et évaluations")

    st.markdown("#### 🧪 Jeux de données testés")
    st.markdown("""
    - **Données synthétiques** (catégories Jouets, Alimentation, Vêtements)
    - **Données M5 Forecasting** : vraie base complexe avec prix, calendrier, ventes quotidiennes
    """)

    st.markdown("#### 🧪 Méthodologie d’évaluation")
    st.markdown("""
    - Séparation **temporelle** des données : train / validation / test
    - Métriques utilisées :
        - **RMSE** : écart-type des résidus
        - **MAPE** : erreur en pourcentage
        - **MAE / MSE**
    - Visualisation des **prédictions vs réels**
    - Courbes de performance multi-granularité
    """)

    st.markdown("#### 📌 Résultats clés")
    st.markdown("""
    - Prophet : robuste sur données agrégées
    - XGBoost : très performant avec bon feature engineering
    - LSTM : efficace mais sensible aux paramètres
    - DeepAR : le plus fiable sur longues séries multi-produits
    """)

    st.success("Un onglet de comparaison dynamique des modèles est disponible dans l'application !")

# Footer simple
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 0.9em; color: gray'>
    © 2025 – Application de Prévision des Ventes – Master DSC
</div>
""", unsafe_allow_html=True)
