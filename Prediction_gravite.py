import pandas as pd

def predict_gravite(df, modele_r6, pipeline_grave):
    """
    df : DataFrame contenant les mêmes colonnes que X_train
    modele_r6 : modèle binaire (grave / non grave)
    pipeline_grave : pipeline pour prédire hospitalisé / tué
    """

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
