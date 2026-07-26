import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

st.title(" Analyse complète : Gravité, Modèles & Feature Importance")

# =============================================================
# Chargement des données
# =============================================================

df = pd.read_csv("data/accidents_France_encoded.zip")

X = df.drop(columns=["gravite"])
y = df["gravite"]

# Binarisation identique à ton notebook
def binariser(y):
    y = y.values.ravel()
    return np.where(np.isin(y, [1, 4]), 0, 1)

y_bin = binariser(y)

x_train, x_test, y_train_bin, y_test_bin = train_test_split(
    X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
)

# =============================================================
# SECTION 1 — Distribution multiclasses + binaire
# =============================================================

st.header(" Distribution des classes")

mapping = {1: "Indemne", 2: "Tué", 3: "Hosp.", 4: "Léger"}
noms_binaire = {
    0: "Non grave\n(Indemne + Léger)",
    1: "Grave\n(Hosp. + Tué)"
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Multi-classes
counts_multi = y.value_counts().sort_index()
sns.barplot(
    x=[mapping[int(i)] for i in counts_multi.index],
    y=counts_multi.values,
    palette="viridis",
    ax=axes[0]
)
axes[0].set_title("Distribution originale — 4 classes")
axes[0].set_ylabel("Nombre d'accidents")

# Binaire
counts_bin = pd.Series(y_bin).value_counts().sort_index()
sns.barplot(
    x=[noms_binaire[int(i)] for i in counts_bin.index],
    y=counts_bin.values,
    palette=["#5DCAA5", "#D85A30"],
    ax=axes[1]
)
axes[1].set_title("Distribution binaire — Grave / Non grave")
axes[1].set_ylabel("Nombre d'accidents")

plt.tight_layout()
st.pyplot(fig)

# =============================================================
# SECTION 2 — Comparaison des modèles
# =============================================================

st.header(" Comparaison des modèles")

def evaluer(nom, modele, x_tr, y_tr, x_te, y_te):
    modele.fit(x_tr, y_tr)
    y_pred = modele.predict(x_te)
    y_proba = modele.predict_proba(x_te)[:, 1]

    acc = accuracy_score(y_te, y_pred)
    f1_macro = f1_score(y_te, y_pred, average="macro")
    auc = roc_auc_score(y_te, y_proba)
    recall_grave = classification_report(
        y_te, y_pred, output_dict=True
    )["1"]["recall"]

    # Matrice de confusion
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_te, y_pred,
        display_labels=["Non grave", "Grave"],
        cmap="Blues", ax=ax
    )
    ax.set_title(f"Matrice de confusion — {nom}")
    st.pyplot(fig)

    return {
        "Modèle": nom,
        "Accuracy": round(acc, 4),
        "Macro F1": round(f1_macro, 4),
        "AUC": round(auc, 4),
        "Recall grave": round(recall_grave, 4)
    }

# Liste des modèles
n_neg = (y_train_bin == 0).sum()
n_pos = (y_train_bin == 1).sum()
scale = n_neg / n_pos

modeles = [
    ("Dummy (most_frequent)", DummyClassifier(strategy="most_frequent")),
    ("Régression logistique (balanced)",
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ("Decision Tree (balanced)",
        DecisionTreeClassifier(class_weight="balanced", max_depth=10, random_state=42)),
    ("XGBoost pondéré",
        XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=0.7,
            min_child_weight=5,
            scale_pos_weight=scale,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )),
    ("Balanced Random Forest optimisé",
        BalancedRandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=2,
            sampling_strategy="not minority",
            random_state=42,
            n_jobs=-1
        ))
]

resultats = []

for nom, modele in modeles:
    st.subheader(f" {nom}")
    r = evaluer(nom, modele, x_train, y_train_bin, x_test, y_test_bin)
    resultats.append(r)

synthese = pd.DataFrame(resultats).sort_values("Recall grave", ascending=False)

st.subheader(" Synthèse comparative")
st.dataframe(synthese)

# Graphique comparatif
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metriques = ["Macro F1", "AUC", "Recall grave"]

for ax, metrique in zip(axes, metriques):
    sns.barplot(
        data=synthese, x=metrique, y="Modèle",
        palette="viridis", ax=ax
    )
    ax.set_xlim(0, 1)
    ax.set_title(metrique)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)

plt.tight_layout()
st.pyplot(fig)

# =============================================================
# SECTION 3 — Feature Importance du modèle r6
# =============================================================

st.header(" Feature Importance — Modèle r6")

# On réentraîne r6 ici (BRF optimisé)
modele_r6 = BalancedRandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=2,
    sampling_strategy="not minority",
    random_state=42,
    n_jobs=-1
)
modele_r6.fit(x_train, y_train_bin)

importances_r6 = modele_r6.feature_importances_
features = X.columns

fi_r6 = pd.DataFrame({
    "feature": features,
    "importance": importances_r6
}).sort_values("importance", ascending=False)

# Graphique
fig, ax = plt.subplots(figsize=(10, 12))
ax.barh(fi_r6["feature"], fi_r6["importance"], color="steelblue")
ax.invert_yaxis()
ax.set_title("Feature Importance — Modèle r6")
plt.tight_layout()
st.pyplot(fig)

# Tableau
st.subheader("Tableau des importances")
st.dataframe(fi_r6)

# Listing texte
st.subheader("Listing complet")
st.text(fi_r6.to_string(index=False))
