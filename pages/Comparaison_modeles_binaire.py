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
from models_loader import modele_r6, pipeline_grave, load_all_models

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

st.title("Comparaison des modèles — Approche binaire")

# =============================================================
# SECTION 1 — Distribution des classes (avant/après binarisation)
# =============================================================

@st.cache_data
def charger_donnees_completes():
    return pd.read_csv("data/accidents_France_encoded.zip")

df_complet = charger_donnees_completes()

st.header("1. Distribution des classes")

MAPPING_GRAVITE = {1: "Indemne", 2: "Tué", 3: "Hosp.", 4: "Léger"}
y_complet = df_complet["gravite"]
y_complet_bin = binariser(nettoyer_cible(y_complet.to_frame()))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

counts_multi = y_complet.value_counts().sort_index()
sns.barplot(
    x=[MAPPING_GRAVITE[int(i)] for i in counts_multi.index],
    y=counts_multi.values,
    palette="viridis",
    ax=axes[0]
)
axes[0].set_title("Distribution originale — 4 classes")

counts_bin = pd.Series(y_complet_bin).value_counts().sort_index()
sns.barplot(
    x=["Non grave", "Grave"],
    y=counts_bin.values,
    palette=["#5DCAA5", "#D85A30"],
    ax=axes[1]
)
axes[1].set_title("Distribution binaire — Grave / Non grave")

plt.tight_layout()
st.pyplot(fig)

st.caption(
    "Le regroupement en 2 classes ('Grave' = Tué + Hospitalisé) rééquilibre "
    "la distribution par rapport aux 4 classes d'origine, où 'Tué' était "
    "fortement minoritaire."
)

st.divider()

# =============================================================
# SECTION 2 — Grave vs Non grave : matrices de confusion combinées
# =============================================================

st.header("2. Grave vs Non grave")
st.caption(
    "Tous les modèles ci-dessous sont évalués sur la même tâche "
    "(distinguer un accident grave d'un accident non grave)."
)

def evaluer_modele(nom, modele, X_eval, y_eval, ax_matrice, labels=("Non grave", "Grave")):
    y_pred = modele.predict(X_eval)
    y_proba = modele.predict_proba(X_eval)[:, 1]

    acc = accuracy_score(y_eval, y_pred)
    f1 = f1_score(y_eval, y_pred)
    auc = roc_auc_score(y_eval, y_proba)
    recall_positif = classification_report(
        y_eval, y_pred, output_dict=True, zero_division=0
    )["1"]["recall"]

    ConfusionMatrixDisplay.from_predictions(
        y_eval, y_pred,
        display_labels=labels,
        cmap="Blues", ax=ax_matrice, colorbar=False
    )
    ax_matrice.set_title(nom, fontsize=10)

    return {
        "Modèle": nom,
        "Accuracy": round(acc, 4),
        "F1-score": round(f1, 4),
        "AUC": round(auc, 4),
        "Recall (classe positive)": round(recall_positif, 4)
    }

with st.spinner("Chargement des modèles de comparaison..."):
    modeles_grave = load_all_models()

fig, axes = plt.subplots(1, len(modeles_grave), figsize=(4 * len(modeles_grave), 4))
if len(modeles_grave) == 1:
    axes = [axes]

resultats_grave = []
for ax, (nom, modele) in zip(axes, modeles_grave.items()):
    res = evaluer_modele(nom, modele, X, y_bin, ax)
    resultats_grave.append(res)

plt.suptitle("Matrices de confusion — Grave vs Non grave", fontsize=13, y=1.05)
plt.tight_layout()
st.pyplot(fig)

# =============================================================
# SECTION 3 — Tableau comparatif des recalls
# =============================================================

st.header("3. Tableau comparatif des recalls")

df_grave = pd.DataFrame(resultats_grave).sort_values(
    "Recall (classe positive)", ascending=False
)

st.dataframe(
    df_grave.style.background_gradient(subset=["Recall (classe positive)"], cmap="Reds")
)

recall_r6 = df_grave.loc[
    df_grave["Modèle"] == "Balanced Random Forest (modele_r6 - retenu)",
    "Recall (classe positive)"
].values[0]

st.markdown(
    f"""
**Modèle retenu : Balanced Random Forest (`modele_r6`)**

On retient ce modèle car il offre le meilleur compromis entre un recall élevé
sur la classe "Grave" ({recall_r6:.1%}) et des performances globales
stables (Accuracy, F1, AUC). Un modèle comme XGBoost peut afficher une
meilleure accuracy globale tout en ratant beaucoup plus de cas graves, ce
qui n'est pas acceptable pour ce cas d'usage — c'est le recall, pas
l'accuracy, qui guide ce choix.
"""
)

st.divider()

# =============================================================
# SECTION 4 — Feature importance du modèle retenu (modele_r6)
# =============================================================

st.header("4. Feature importance — Modèle retenu (modele_r6)")

fi = pd.DataFrame({
    "feature": X.columns,
    "importance": modele_r6.feature_importances_
}).sort_values("importance", ascending=False)

fig, ax = plt.subplots(figsize=(4, 5))
ax.barh(fi["feature"], fi["importance"], color="steelblue")
ax.invert_yaxis()
ax.set_title("Feature Importance — Modèle r6", fontsize=9)
ax.tick_params(axis="y", labelsize=5)
ax.tick_params(axis="x", labelsize=6)
plt.tight_layout()
st.pyplot(fig)

with st.expander("Voir le tableau complet des importances"):
    st.dataframe(fi)

st.divider()

# =============================================================
# SECTION 5 — Pipeline grave : Hospitalisé vs Tué
# =============================================================
# IMPORTANT : pipeline_grave a été entraîné UNIQUEMENT sur le sous-ensemble
# des accidents déjà classés "graves" (hospitalisé=3 ou tué=2), pas sur tout
# X_test. On ne peut donc pas le comparer aux modèles ci-dessus, ni l'évaluer
# avec y_bin (grave/non grave) : ce serait comparer deux tâches différentes.
# On l'évalue ici sur sa vraie tâche, sur le bon sous-ensemble.

st.header("5. Hospitalisé vs Tué (parmi les accidents graves uniquement)")
st.caption(
    "Ce modèle ne prédit rien sur les accidents non graves : il s'applique "
    "uniquement aux cas déjà identifiés comme graves, pour distinguer "
    "hospitalisation et décès. Il n'est pas comparable au tableau ci-dessus."
)

mask_graves = y_clean.isin([2, 3])
X_graves = X.loc[mask_graves]
y_graves = y_clean.loc[mask_graves].map({2: 1, 3: 0})  # 1 = Tué, 0 = Hospitalisé

fig, ax = plt.subplots(figsize=(5, 4))
res_pipeline = evaluer_modele(
    "Pipeline Grave (Hospitalisé vs Tué)",
    pipeline_grave,
    X_graves, y_graves, ax,
    labels=("Hospitalisé", "Tué")
)
st.pyplot(fig)

st.subheader("Résultat — Pipeline Grave")
st.dataframe(pd.DataFrame([res_pipeline]))
