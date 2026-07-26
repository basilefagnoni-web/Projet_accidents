import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import os
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    ConfusionMatrixDisplay
)

# ============================================================
# Chargement des données (cache)
# ============================================================

DATA_DIR = "data"

@st.cache_data
def charger_test():
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.zip"))
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.zip"))
    return X_test, y_test

X, y = charger_test()

def binariser(y):
    return np.where(np.isin(y, [1, 4]), 0, 1)

y_bin = binariser(y.values.ravel())


# ============================================================
# Téléchargement Drive uniquement pour modele_r6.pkl
# ============================================================

drive_urls = {
    "Balanced Random Forest": "https://drive.google.com/uc?id=1Ogz0_I2gGtWgRaeeU0JkapgROe5lYcR5"
}

@st.cache_resource
def charger_modele(path, nom):
    # Cas particulier : modèle r6 sur Google Drive
    if nom == "Balanced Random Forest":
        url = drive_urls[nom]
        os.makedirs("models", exist_ok=True)
        if not os.path.exists(path):
            gdown.download(url, path, quiet=False)
        return joblib.load(path)

    # Cas général : modèles locaux
    return joblib.load(path)

# ============================================================
# Définition des modèles pré‑entraînés
# ============================================================

model_paths = {
    "Dummy": "models/modele_dummy.pkl",
    "Régression logistique": "models/modele_lr.pkl",
    "Decision Tree": "models/modele_dt.pkl",
    "XGBoost": "models/modele_xgb.pkl",
    "Balanced Random Forest": "models/modele_r6.pkl"   # r6 = BRF optimisé
}

# ============================================================
# Interface Streamlit
# ============================================================

st.title("Comparaison des modèles pré‑entraînés")

resultats = []

for nom, path in model_paths.items():
    st.subheader(f"{nom}")

    modele = charger_modele(path, nom)
    y_pred = modele.predict(X)
    y_proba = modele.predict_proba(X)[:, 1]

    acc = accuracy_score(y_bin, y_pred)
f1 = f1_score(y_bin, y_pred)
auc = roc_auc_score(y_bin, y_proba)

# Calcul du recall grave (classe 1)
recall_grave = classification_report(
    y_bin, y_pred, output_dict=True
)["1"]["recall"]

resultats.append({
    "Modèle": nom,
    "Accuracy": acc,
    "F1-score": f1,
    "AUC": auc,
    "Recall grave": recall_grave
})


    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_bin, y_pred,
        display_labels=["Non grave", "Grave"],
        cmap="Blues",
        ax=ax
    )
    ax.set_title(f"Matrice de confusion — {nom}")
    st.pyplot(fig)

df_scores = pd.DataFrame(resultats)
st.subheader("Tableau comparatif")
st.dataframe(df_scores)

fig, ax = plt.subplots(figsize=(10, 5))
df_scores.set_index("Modèle")[["Accuracy", "F1-score", "AUC"]].plot(
    kind="bar",
    ax=ax,
    color=["#3498db", "#e67e22", "#2ecc71"]
)
ax.set_ylim(0, 1)
ax.set_title("Comparaison des métriques")
st.pyplot(fig)

# ============================================================
# Synthèse comparative (triée sur Recall grave)
# ============================================================

st.subheader("Synthèse comparative — Modèle binaire grave / non grave")

# Tri sur Recall grave (comme dans ton script d'origine)
synthese = df_scores.sort_values("Recall grave", ascending=False)

st.write("Classement des modèles (du meilleur Recall grave au moins bon) :")
st.dataframe(synthese)

# Visualisation synthèse
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
metriques = ["Accuracy", "F1-score", "AUC", "Recall grave"]

for ax, metrique in zip(axes, metriques):
    sns.barplot(
        data=synthese, x=metrique, y="Modèle",
        palette="viridis", ax=ax
    )
    ax.set_xlim(0, 1)
    ax.set_title(metrique)
    ax.set_xlabel("")
    ax.set_ylabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)

plt.tight_layout()
st.pyplot(fig)


