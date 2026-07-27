import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    ConfusionMatrixDisplay, classification_report
)

# On importe les modèles déjà chargés dans streamlit_app.py
from streamlit_app import modele_r6, pipeline_grave

DATA_DIR = "data"

@st.cache_data
def charger_test():
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.zip")
    y_test = pd.read_csv(f"{DATA_DIR}/y_test.zip")
    return X_test, y_test

X, y = charger_test()

def binariser(y):
    return np.where(np.isin(y, [1, 4]), 0, 1)

y_bin = binariser(y.values.ravel())

# On garde les mêmes noms de modèles
modeles = {
    "Balanced Random Forest": modele_r6,
    "Pipeline Grave": pipeline_grave
}

st.title("Comparaison des modèles pré‑entraînés")

resultats = []

for nom, modele in modeles.items():
    st.subheader(f"{nom}")

    y_pred = modele.predict(X)
    y_proba = modele.predict_proba(X)[:, 1]

    acc = accuracy_score(y_bin, y_pred)
    f1 = f1_score(y_bin, y_pred)
    auc = roc_auc_score(y_bin, y_proba)

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

st.subheader("Synthèse comparative — Modèle binaire grave / non grave")

synthese = df_scores.sort_values("Recall grave", ascending=False)
st.dataframe(synthese)

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
