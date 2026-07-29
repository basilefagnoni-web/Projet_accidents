import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title(" Analyse des données d'accidents")

# =============================================================
# Chargement des données
# =============================================================

df = pd.read_csv("data/accidents_France_encoded.zip")

st.write("Aperçu des données :")
st.dataframe(df.head())

st.subheader("Statistiques générales")
st.write(df.describe())

# =============================================================
# 1. Distribution de la gravité (variable cible)
# =============================================================

st.header("Distribution de la gravité")

GRAVITE_LABELS = {1: "Indemne", 2: "Tué", 3: "Hospitalisé", 4: "Blessé léger"}
repartition_gravite = df["gravite"].map(GRAVITE_LABELS).value_counts()

fig, ax = plt.subplots(figsize=(8, 4))
repartition_gravite.plot(kind="bar", ax=ax, color="#2a78d6")
ax.set_ylabel("Nombre d'accidents")
ax.set_xlabel("")
ax.set_title("Répartition des usagers par niveau de gravité")
plt.xticks(rotation=0)
st.pyplot(fig)

st.caption(
    f"Sur {len(df):,} lignes, la classe la plus fréquente est "
    f"« {repartition_gravite.idxmax()} » ({repartition_gravite.max():,} occurrences)."
)

# =============================================================
# 2. Distribution de la vitesse maximale autorisée
# =============================================================

st.header("Distribution de la vitesse maximale autorisée")

fig, ax = plt.subplots(figsize=(8, 4))
df["vitesse_max"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#eb6834")
ax.set_ylabel("Nombre d'accidents")
ax.set_xlabel("Vitesse maximale autorisée (km/h)")
plt.xticks(rotation=0)
st.pyplot(fig)

# =============================================================
# 3. Distribution de l'âge des usagers
# =============================================================

st.header("Distribution de l'âge des usagers")

fig, ax = plt.subplots(figsize=(8, 4))
df["age_usager"].plot(kind="hist", bins=30, ax=ax, color="#1baf7a")
ax.set_xlabel("Âge de l'usager")
ax.set_ylabel("Nombre d'usagers")
st.pyplot(fig)

# =============================================================
# 4. Localisation : agglomération vs hors agglomération
# =============================================================

st.header("Localisation des accidents")

AGGLO_LABELS = {0: "Hors agglomération", 1: "Agglomération"}
repartition_agglo = df["localisation_agglo"].map(AGGLO_LABELS).value_counts()

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(repartition_agglo, labels=repartition_agglo.index, autopct="%1.0f%%",
           colors=["#4a3aa7", "#e34948"])
    st.pyplot(fig)
with col2:
    st.metric("Agglomération", f"{repartition_agglo.get('Agglomération', 0):,}")
    st.metric("Hors agglomération", f"{repartition_agglo.get('Hors agglomération', 0):,}")

# =============================================================
# 5. Répartition par sexe de l'usager
# =============================================================

st.header("Répartition par sexe")

SEXE_LABELS = {1: "Homme", 2: "Femme"}
repartition_sexe = df["sexe_usager"].map(SEXE_LABELS).value_counts()

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(repartition_sexe.index, repartition_sexe.values, color=["#378add", "#d4537e"])
ax.set_ylabel("Nombre d'usagers")
st.pyplot(fig)

# =============================================================
# 6. Répartition par tranche horaire
# =============================================================

st.header("Répartition par tranche horaire")

TRANCHE_LABELS = {1: "Matin", 2: "Après-midi", 3: "Soir", 4: "Nuit"}
repartition_horaire = df["tranche_horaire_ord"].map(TRANCHE_LABELS).value_counts().reindex(
    ["Matin", "Après-midi", "Soir", "Nuit"]
)

fig, ax = plt.subplots(figsize=(6, 4))
repartition_horaire.plot(kind="bar", ax=ax, color="#eda100")
ax.set_ylabel("Nombre d'accidents")
ax.set_xlabel("")
plt.xticks(rotation=0)
st.pyplot(fig)

# =============================================================
# 7. Répartition week-end / jour férié
# =============================================================

st.header("Week-end et jours fériés")

col1, col2 = st.columns(2)
with col1:
    st.write("Accidents le week-end vs en semaine")
    fig, ax = plt.subplots(figsize=(4, 3))
    df["is_weekend"].map({0: "Semaine", 1: "Week-end"}).value_counts().plot(
        kind="bar", ax=ax, color="#1baf7a"
    )
    plt.xticks(rotation=0)
    st.pyplot(fig)
with col2:
    st.write("Accidents un jour férié vs un jour normal")
    fig, ax = plt.subplots(figsize=(4, 3))
    df["is_ferie"].map({0: "Jour normal", 1: "Jour férié"}).value_counts().plot(
        kind="bar", ax=ax, color="#e87ba4"
    )
    plt.xticks(rotation=0)
    st.pyplot(fig)
