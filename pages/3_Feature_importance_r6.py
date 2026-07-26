import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import gdown
import os

# ============================================================
# Chargement des données test (cache)
# ============================================================

@st.cache_data
def charger_donnees_test():
    return pd.read_csv("data/X_test.zip")

# ============================================================
# Téléchargement Drive + cache pour modele_r6.pkl
# ============================================================

@st.cache_resource
def charger_modele_r6():
    url_r6 = "https://drive.google.com/uc?id=1Ogz0_I2gGtWgRaeeU0JkapgROe5lYcR5"
    local_path = "models/modele_r6.pkl"

    os.makedirs("models", exist_ok=True)

    if not os.path.exists(local_path):
        gdown.download(url_r6, local_path, quiet=False)

    return joblib.load(local_path)

# ============================================================
# Chargement des données + modèle
# ============================================================

X = charger_donnees_test()
modele_r6 = charger_modele_r6()

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
