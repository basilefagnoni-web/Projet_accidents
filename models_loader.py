import os
import joblib
import gdown
import streamlit as st

# =============================================================
# Modèles principaux (utilisés pour la prédiction) — via Google Drive
# =============================================================

MODEL_URL = "https://drive.google.com/uc?id=1BEWt3Aphz-_xiftmqVlty18rRMNTvIcM"
PIPELINE_URL = "https://drive.google.com/uc?id=1ztGBGcuegiJ3BvsPnUrXo-x4449C9tjr"

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "modele_r6.pkl")
PIPELINE_PATH = os.path.join(MODELS_DIR, "pipeline_grave.pkl")

def download_models():
    """Télécharge les modèles principaux uniquement s'ils n'existent pas déjà."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    if not os.path.exists(PIPELINE_PATH):
        gdown.download(PIPELINE_URL, PIPELINE_PATH, quiet=False)

@st.cache_resource
def load_models():
    """
    Charge les modèles principaux UNE SEULE FOIS.
    Utilisés sur la page de prédiction. Chargés au démarrage de l'app.
    """
    download_models()
    modele_r6 = joblib.load(MODEL_PATH)
    pipeline_grave = joblib.load(PIPELINE_PATH)
    return modele_r6, pipeline_grave

# Chargement effectif des modèles principaux (une seule fois, au démarrage)
modele_r6, pipeline_grave = load_models()


# =============================================================
# Modèles de comparaison (Dummy, LR, DT, XGB) — déjà commités dans le repo
# Chargés SEULEMENT à la demande (page comparaison), pas au démarrage de l'app,
# pour ne pas alourdir la RAM des autres pages qui n'en ont pas besoin.
# =============================================================

DUMMY_PATH = os.path.join(MODELS_DIR, "modele_dummy.pkl")
LR_PATH = os.path.join(MODELS_DIR, "modele_lr.pkl")
DT_PATH = os.path.join(MODELS_DIR, "modele_dt.pkl")
XGB_PATH = os.path.join(MODELS_DIR, "modele_xgb.pkl")

@st.cache_resource
def load_comparison_models():
    """
    Charge les 4 modèles de comparaison UNE SEULE FOIS,
    uniquement quand cette fonction est appelée (donc seulement
    depuis la page Comparaison, pas au démarrage global de l'app).
    """
    modele_dummy = joblib.load(DUMMY_PATH)
    modele_lr = joblib.load(LR_PATH)
    modele_dt = joblib.load(DT_PATH)
    modele_xgb = joblib.load(XGB_PATH)
    return modele_dummy, modele_lr, modele_dt, modele_xgb
