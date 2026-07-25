import os
import joblib
import pandas as pd

BASE_DIR = os.getcwd()

MODEL_BIN_PATH = os.path.join(BASE_DIR, "models", "modele_r6.pkl")
MODEL_GRAVE_PATH = os.path.join(BASE_DIR, "models", "pipeline_grave.pkl")

modele_r6 = joblib.load(MODEL_BIN_PATH)
pipeline_grave = joblib.load(MODEL_GRAVE_PATH)

def predict_gravite(df):

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
