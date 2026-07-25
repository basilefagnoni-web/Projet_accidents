import streamlit as st
import os
import gdown
import pandas as pd
import joblib

from Generation_accident import (
    generer_accident,
    SECU_MAPPING,
    COLLISION_MAPPING,
    MANOEUVRE_MAPPING,
    OBSM_MAPPING
)

from Prediction_gravite import predict_gravite


# -----------------------------
# 1) Téléchargement des modèles
# -----------------------------
MODEL_URL = "https://drive.google.com/uc?id=1Ogz0_I2gGtWgRaeeU0JkapgROe5lYcR5"
PIPELINE_URL = "https://drive.google.com/uc?id=1yxYFA2fVwvSc-epAMw50POyz7x0kse0E"

def download_models():
    os.makedirs("models", exist_ok=True)

    if not os.path.exists("models/modele_r6.pkl"):
        gdown.download(MODEL_URL, "models/modele_r6.pkl", quiet=False)

    if not os.path.exists("models/pipeline_grave.pkl"):
        gdown.download(PIPELINE_URL, "models/pipeline_grave.pkl", quiet=False)


# -----------------------------
# 2) Chargement des modèles (cache)
# -----------------------------
@st.cache_resource
def load_models():
    download_models()
    modele_r6 = joblib.load("models/modele_r6.pkl")
    pipeline_grave = joblib.load("models/pipeline_grave.pkl")
    return modele_r6, pipeline_grave


modele_r6, pipeline_grave = load_models()


# -----------------------------
# 3) Interface Streamlit
# -----------------------------
st.title("🚗 Prédiction de gravité d'accident")

secu = st.selectbox("Sécurité", list(SECU_MAPPING.keys()))
collision = st.selectbox("Collision", list(COLLISION_MAPPING.keys()))
manoeuvre = st.selectbox("Manœuvre", list(MANOEUVRE_MAPPING.keys()))
obstacle_mobile = st.selectbox("Obstacle mobile", list(OBSM_MAPPING.keys()))
age = st.slider("Âge", 14, 99, 40)

if st.button("Prédire"):
    # Générer un accident → dictionnaire
    accident_dict = generer_accident(secu, collision, age, manoeuvre, obstacle_mobile)

    # Convertir en DataFrame pour le modèle
    df_accident = pd.DataFrame([accident_dict])

    # Prédire
    result = predict_gravite(df_accident, modele_r6, pipeline_grave)

    st.subheader("Résultat de la prédiction")
    st.write("Gravité :", result["gravite_binaire"][0])
    st.write("Détail :", result["gravite_detaillee"][0])
