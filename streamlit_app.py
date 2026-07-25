import streamlit as st

from generator import (
    generer_accident,
    SECU_MAPPING,
    COLLISION_MAPPING,
    MANOEUVRE_MAPPING,
    OBSM_MAPPING
)

from predict import predict_gravite

import gdown
import os

# Télécharger les modèles
os.makedirs("models", exist_ok=True)

gdown.download(
    "https://drive.google.com/uc?id=1Ogz0_I2gGtWgRaeeU0JkapgROe5lYcR5",
    "models/modele_r6.pkl",
    quiet=False
)

gdown.download(
    "https://drive.google.com/uc?id=1yxYFA2fVwvSc-epAMw50POyz7x0kse0E",
    "models/pipeline_grave.pkl",
    quiet=False
)

st.title("Prédiction de gravité d'accident")

secu = st.selectbox("Sécurité", list(SECU_MAPPING.keys()))
collision = st.selectbox("Collision", list(COLLISION_MAPPING.keys()))
manoeuvre = st.selectbox("Manœuvre", list(MANOEUVRE_MAPPING.keys()))
obstacle_mobile = st.selectbox("Obstacle mobile", list(OBSM_MAPPING.keys()))
age = st.slider("Âge", 14, 99, 40)

if st.button("Prédire"):
    accident = generer_accident(secu, collision, age, manoeuvre, obstacle_mobile)
    result = predict_gravite(accident)

    st.write("Gravité :", result["gravite_binaire"][0])
    st.write("Détail :", result["gravite_detaillee"][0])
