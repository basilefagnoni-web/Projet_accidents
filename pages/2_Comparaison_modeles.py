import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    ConfusionMatrixDisplay, classification_report
)

# On importe les modèles déjà chargés dans models_loader.py
from models_loader import load_all_models, pipeline_grave

DATA_DIR = "data"

# =============================================================
# Chargement des données de test (mis en cache)
# =============================================================

@st.cache_data
def charger_test():
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.zip")
    y_test = pd.read_csv(f"{DATA_DIR}/y_test.zip")
    return X_test, y_test

X, y = charger_test()

def nettoyer_cible(y):
    col = y.iloc[:, 0]
    col = col.astype(str).str.strip().replace("", np.nan)
    col = col.astype(float).astype("Int64")
    return col

def binariser(y_clean):
    return np.where(np.isin(y_clean.values, [1, 4]), 0, 1)

y_clean = nettoyer_cible(y)
y_bin = binariser(y_clean)

st.title("Comparaison des modèles pré-entraînés")

# =============================================================
# Fonction d'évaluation générique (ré-utilisable pour tout modèle déjà entraîné)
# =============================================================

def evaluer_modele(nom, modele, X_eval, y_eval, labels):
    y_pred = modele.predict(X_eval)
    y_proba = modele.predict_proba(X_eval)[:, 1]

    acc = accuracy_score(y_eval, y_pred)
    f1 = f1_score(y_eval, y_pred)
    auc = roc_auc_score(y_eval, y_proba)
    recall_positif = classification_report(
        y_eval, y_pred, output_dict=True, zero_division=0
    )["1"]["recall"]

    st.subheader(nom)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_eval, y_pred,
        display_labels=labels,
        cmap="Blues",
        ax=ax
    )
    ax.set_title(f"Matrice de confusion — {nom}")
    st.pyplot(fig)

    return {
        "Modèle": nom,
        "Accuracy": acc,
        "F1-score": f1,
        "AUC": auc,
        "Recall (classe positive)": recall_positif
    }

# =============================================================
# SECTION 1 — Modèles binaires : Grave vs Non grave
# =============================================================
# Tous ces modèles ont été entraînés pour la même tâche (grave/non grave),
# on peut donc légitimement les comparer entre eux sur le même X_test / y_bin.

st.header("1. Grave vs Non grave")
st.caption(
    "Tous les modèles ci-dessous sont évalués sur la même tâche "
    "(distinguer un accident grave d'un accident non grave)."
)

with st.spinner("Chargement des modèles de comparaison..."):
    modele_dummy, modele_lr, modele_dt, modele_xgb = load_comparison_models()

modeles_grave = {
    "Dummy (most_frequent)": modele_dummy,
    "Régression logistique": modele_lr,
    "Decision Tree": modele_dt,
    "XGBoost": modele_xgb,
    "Balanced Random Forest (modele_r6 - retenu)": modele_r6,
}

resultats_grave = []
for nom, modele in modeles_grave.items():
    res = evaluer_modele(nom, modele, X, y_bin, labels=["Non grave", "Grave"])
    resultats_grave.append(res)

df_grave = pd.DataFrame(resultats_grave).sort_values(
    "Recall (classe positive)", ascending=False
)

st.subheader("Tableau comparatif — Grave vs Non grave")
st.dataframe(df_grave)

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
metriques = ["Accuracy", "F1-score", "AUC", "Recall (classe positive)"]
for ax, metrique in zip(axes, metriques):
    sns.barplot(data=df_grave, x=metrique, y="Modèle", palette="viridis", ax=ax)
    ax.set_xlim(0, 1)
    ax.set_title(metrique)
    ax.set_xlabel("")
    ax.set_ylabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)
plt.tight_layout()
st.pyplot(fig)

# =============================================================
# SECTION 2 — Pipeline grave : Hospitalisé vs Tué
# =============================================================
# IMPORTANT : pipeline_grave a été entraîné UNIQUEMENT sur le sous-ensemble
# des accidents déjà classés "graves" (hospitalisé=3 ou tué=2), pas sur tout
# X_test. On ne peut donc pas le comparer aux modèles ci-dessus, ni l'évaluer
# avec y_bin (grave/non grave) : ce serait comparer deux tâches différentes.
# On l'évalue ici sur sa vraie tâche, sur le bon sous-ensemble.

st.header("2. Hospitalisé vs Tué (parmi les accidents graves uniquement)")
st.caption(
    "Ce modèle ne prédit rien sur les accidents non graves : il s'applique "
    "uniquement aux cas déjà identifiés comme graves, pour distinguer "
    "hospitalisation et décès. Il n'est pas comparable au tableau ci-dessus."
)

mask_graves = y_clean.isin([2, 3])
X_graves = X.loc[mask_graves]
y_graves = y_clean.loc[mask_graves].map({2: 1, 3: 0})  # 1 = Tué, 0 = Hospitalisé

res_pipeline = evaluer_modele(
    "Pipeline Grave (Hospitalisé vs Tué)",
    pipeline_grave,
    X_graves, y_graves,
    labels=["Hospitalisé", "Tué"]
)

st.subheader("Résultat — Pipeline Grave")
st.dataframe(pd.DataFrame([res_pipeline]))
