import joblib
import pandas as pd

MODEL_BIN_PATH = "models/modele_r6.pkl"
MODEL_GRAVE_PATH = "models/pipeline_grave.pkl"

# Charger les modèles
modele_r6 = joblib.load(MODEL_BIN_PATH)
pipeline_grave = joblib.load(MODEL_GRAVE_PATH)

def predict_gravite(df):

    # 1. Prédiction grave / non grave
    proba_grave = modele_r6.predict_proba(df)[:, 1]
    pred_grave = (proba_grave >= 0.5).astype(int)

    resultats = []

    for i, is_grave in enumerate(pred_grave):

        if is_grave == 0:
            resultats.append({
                "gravite_binaire": "Non grave",
                "gravite_detaillee": "Indemne ou Léger"
            })

        else:
            ligne = df.iloc[[i]]
            pred_ht = pipeline_grave.predict(ligne)[0]
            detail = "Hospitalisé" if pred_ht == 0 else "Tué"

            resultats.append({
                "gravite_binaire": "Grave",
                "gravite_detaillee": detail
            })

    return pd.DataFrame(resultats)
