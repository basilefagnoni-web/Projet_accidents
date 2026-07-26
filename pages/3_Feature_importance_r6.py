import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

@st.cache_data
def charger_donnees():
    return pd.read_csv("data/accidents_France_encoded.zip")

@st.cache_resource
def charger_modele():
    return joblib.load("models/modele_r6.pkl")

df = charger_donnees()
modele_r6 = charger_modele()

X = df.drop(columns=["gravite"])
features = X.columns
importances = modele_r6.feature_importances_

fi = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values("importance", ascending=False)

st.title("📈 Feature Importance — Modèle r6")

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
