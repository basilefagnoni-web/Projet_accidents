import os
import joblib
import gdown
import streamlit as st

# =============================================================
# Dossier des modèles
# =============================================================

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# =============================================================
# Modèles Drive (r6 + pipeline_grave)
# =============================================================

MODEL_URL = "https://drive.google.com/uc?id=1BEWt3Aphz-_xiftmqVlty18rRMNTvIcM"
PIPELINE_URL = "https://drive.google.com/uc?id=1ztGBGcuegiJ3BvsPnUrXo-x4449C9tjr"

MODEL_PATH = os.path.join(MODELS_DIR, "modele_r6.pkl")
PIPELINE_PATH = os.path.join(MODELS_DIR, "pipeline_grave.pkl")

def download_models():
    """Télécharge les modèles Drive uniquement s'ils n'existent pas."""
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    if not os.path.exists(PIPELINE_PATH):
        gdown.download(PIPELINE_URL, PIPELINE_PATH, quiet=False)

@st.cache_resource
def load_drive_models():
    """
    Charge les modèles Drive (Balanced Random Forest + Pipeline Grave)
    une seule fois.
    """
    download_models()
    modele_r6 = joblib.load(MODEL_PATH)
    pipeline_grave = joblib.load(PIPELINE_PATH)
    return modele_r6, pipeline_grave

# Chargement au démarrage : disponibles partout dans l'app
modele_r6, pipeline_grave = load_drive_models()

# =============================================================
# Modèles locaux (Dummy, LR, DT, XGB)
# =============================================================

LOCAL_MODELS = {
    "Dummy (most_frequent)": os.path.join(MODELS_DIR, "modele_dummy.pkl"),
    "Régression logistique": os.path.join(MODELS_DIR, "modele_lr.pkl"),
    "Decision Tree": os.path.join(MODELS_DIR, "modele_dt.pkl"),
    "XGBoost": os.path.join(MODELS_DIR, "modele_xgb.pkl")
}

@st.cache_resource
def load_local_models():
    """Charge les modèles locaux commités dans le repo GitHub."""
    loaded = {}
    for name, path in LOCAL_MODELS.items():
        loaded[name] = joblib.load(path)
    return loaded

# =============================================================
# Fonction unifiée pour la comparaison (Grave vs Non grave)
# =============================================================

def load_all_models():
    """
    Renvoie un dictionnaire de modèles pour la comparaison binaire
    Grave vs Non grave (page 2) :

    - Dummy
    - Régression logistique
    - Decision Tree
    - XGBoost
    - Balanced Random Forest (modele_r6 - retenu)

    Pipeline Grave est volontairement exclu de cette comparaison,
    mais reste disponible via la variable globale `pipeline_grave`
    pour la section Hospitalisé vs Tué.
    """
    local = load_local_models()
    return {
        **local,
        "Balanced Random Forest (modele_r6 - retenu)": modele_r6
    }
