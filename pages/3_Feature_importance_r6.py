import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# On importe le modèle déjà chargé dans models_loader.py
from models_loader import modele_r6

@st.cache_data
def charger_donnees_test():
    return pd.read_csv("data/X_test.zip")

X = charger_donnees_test()

features = X.columns
importances = modele_r6.feature_importances_

fi = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values("importance", ascending=False)

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
