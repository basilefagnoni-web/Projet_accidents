import streamlit as st
import os
import gdown
import joblib
import pandas as pd
import numpy as np
from Generation_accident import (
    generer_accident,
    SECU_MAPPING,
    COLLISION_MAPPING,
    MANOEUVRE_MAPPING,
    OBSM_MAPPING,
    OBS_FIXE_MAPPING
)
from Prediction_gravite import predict_gravite

# -----------------------------
# Chargement des modèles
# -----------------------------
from models_loader import modele_r6, pipeline_grave

# -----------------------------
# Interface Streamlit
# -----------------------------
st.title(" Prédiction de gravité d'accident")
st.write("Paramètres influençant la gravité : sécurité, collision, manœuvre, obstacle mobile, obstacle fixe, vitesse, localisation et âge.")

# Paramètres utilisateur
secu = st.selectbox("Sécurité", list(SECU_MAPPING.keys()))
collision = st.selectbox("Collision", list(COLLISION_MAPPING.keys()))
manoeuvre = st.selectbox("Manœuvre", list(MANOEUVRE_MAPPING.keys()))
obstacle_mobile = st.selectbox("Obstacle mobile", list(OBSM_MAPPING.keys()))
obs_fixe = st.selectbox("Obstacle fixe heurté", list(OBS_FIXE_MAPPING.keys()))
age = st.slider("Âge de l'usager", 14, 99, 40)
vitesse = st.slider("Vitesse maximale (km/h)", 10, 150, 50)
agglo = st.selectbox("Localisation", ["Agglomération", "Hors agglomération"])
agglo_val = 1 if agglo == "Agglomération" else 0

if st.button("Prédire"):
    df_accident = generer_accident(
        secu=secu,
        collision=collision,
        age=age,
        manoeuvre=manoeuvre,
        obstacle_mobile=obstacle_mobile,
        vitesse_max=vitesse,
        obs_fixe=obs_fixe
    )
    df_accident["localisation_agglo"] = agglo_val

    result = predict_gravite(df_accident, modele_r6, pipeline_grave)
    st.subheader("Résultat de la prédiction")
    st.write("Gravité :", result["gravite_binaire"][0])
    st.write("Détail :", result["gravite_detaillee"][0])
