import os
import joblib
import gdown
import streamlit as st

# URLs Drive des modèles
MODEL_URL = "https://drive.google.com/uc?id=1BEWt3Aphz-_xiftmqVlty18rRMNTvIcM"
PIPELINE_URL = "https://drive.google.com/uc?id=1ztGBGcuegiJ3BvsPnUrXo-x4449C9tjr"

# Dossier local où stocker les modèles
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "modele_r6.pkl")
PIPELINE_PATH = os.path.join(MODELS_DIR, "pipeline_grave.pkl")


def download_models():
    """Télécharge les modèles uniquement s'ils n'existent pas déjà."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    if not os.path.exists(PIPELINE_PATH):
        gdown.download(PIPELINE_URL, PIPELINE_PATH, quiet=False)


@st.cache_resource
def load_models():
    """
    Charge les modèles UNE SEULE FOIS.
    Cette fonction est exécutée hors du thread principal,
    ce qui évite le deadlock Uvicorn.
    """
    download_models()
    modele_r6 = joblib.load(MODEL_PATH)
    pipeline_grave = joblib.load(PIPELINE_PATH)
    return modele_r6, pipeline_grave


# Chargement effectif des modèles (une seule fois)
modele_r6, pipeline_grave = load_models()
