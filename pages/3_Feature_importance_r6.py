import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import gdown
import os

# ============================================================
# Chargement des données (cache)
# ============================================================

@st.cache_data
def charger_donnees():
    return pd.read_csv("data/accidents_France_encoded.zip")

# ============================================================
# Téléchargement Drive + cache pour modele_r6.pkl
# ============================================================

@st.cache_resource
def charger_modele_r6():
    # URL Google Drive (remplace ID_R6 par ton vrai ID)
    url_r6 = "https://drive.google.com/uc?id=ID_R6"

    # Chemin local du modèle
    local_path = "models/modele_r6.pkl"

    # Création du dossier si nécessaire
    os.makedirs("models", exist_ok=True)

    # Téléchargement uniquement si le fichier n'existe pas
    if not os.path.exists(local_path):
        gdown.download(url_r6, local_path, quiet=False)

    # Chargement du modèle
    return joblib.load(local_path)

# ============================================================
# Chargement des données + modèle
# ============================================================

df = charger_donnees()
modele_r6 = charger_modele_r6()

X = df.drop(columns=["gravite"])
features = X.columns
importances = modele_r6.feature_importances_

fi = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values("importance", ascending=False)

# ============================================================
# Interface Streamlit
# ============================================================

st.title("Feature Importance — Modèle r6")

fig, ax = plt.subplots(figsize=(10, 12))
ax.barh(fi["feature"], fi["importance"], color="steelblue")
ax.invert_yaxis()
ax.set_title("Feature Importance — Modèle r6")
plt.tight_layout()
st.pyplot(fig)

st.subheader("Tableau des importances")
st.dataframe(fi)

st.subheader("Listing complet")
st.text(fi.to_string(index=False))
