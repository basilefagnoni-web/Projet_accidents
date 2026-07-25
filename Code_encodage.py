import os
import pandas as pd
import category_encoders as ce
from sklearn.model_selection import train_test_split

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def categoriser_heure(h):
    if 6 <= h < 12:
        return "Matin"
    elif 12 <= h < 18:
        return "Apres-midi"
    elif 18 <= h < 22:
        return "Soir"
    return "Nuit"

def run_encoding_pipeline():
    df = pd.read_csv(os.path.join(DATA_DIR, "accidents_France_raw.zip"))

    # Tranche horaire
    heure = df["hrmn"].astype(str).str.replace(":", "").str.zfill(4).str[:2].astype(int)
    df["TRANCHE_HORAIRE"] = heure.apply(categoriser_heure)
    df["tranche_horaire_ord"] = df["TRANCHE_HORAIRE"].map(
        {"Matin":1,"Apres-midi":2,"Soir":3,"Nuit":4}
    )

    # Colonnes inutiles
    df.drop(columns=[
        "jour","mois","date_time","hrmn","an",
        "equipement_secu_3","action_pieton",
        "lat","long"
    ], inplace=True, errors="ignore")

    # Nettoyage
    df = df.dropna(subset=["type_collision","luminosite","meteo"])
    df["equipement_secu_2"] = df["equipement_secu_2"].fillna(0)

    # Split
    X = df.drop(columns=["gravite"])
    y = df["gravite"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    df_train = X_train.copy()
    df_train["gravite"] = y_train
    df_test = X_test.copy()
    df_test["gravite"] = y_test

    # Imputation mode
    cols_mode = ["regime_circulation","motif_deplacement","sexe_usager"]
    modes = {col: df_train[col].mode()[0] for col in cols_mode}
    for col in cols_mode:
        df_train[col] = df_train[col].fillna(modes[col])
        df_test[col] = df_test[col].fillna(modes[col])

    # Nettoyage final
    df_train = df_train.dropna(subset=["equipement_secu_1","nb_voies","amenagement"])
    df_test = df_test.dropna(subset=["equipement_secu_1","nb_voies","amenagement"])

    # Regroupement véhicules
    mapping_veh = {
        7:"Voiture",10:"Poids_Lourd_Utilitaire",13:"Poids_Lourd_Utilitaire",
        14:"Poids_Lourd_Utilitaire",17:"Poids_Lourd_Utilitaire",
        1:"Velo_Trotinette",80:"Velo_Trotinette",50:"Velo_Trotinette",
        2:"Deux_Roues_Moteur",30:"Deux_Roues_Moteur",33:"Deux_Roues_Moteur",
        32:"Deux_Roues_Moteur",
        37:"Transport_Commun",38:"Transport_Commun"
    }
    df_train["type_vehicule_simplifie"] = df_train["type_vehicule"].map(mapping_veh).fillna("Autre")
    df_test["type_vehicule_simplifie"] = df_test["type_vehicule"].map(mapping_veh).fillna("Autre")

    # Label encoding
    df_train["sexe_encoded"] = df_train["sexe_usager"].map({1:0,2:1})
    df_test["sexe_encoded"] = df_test["sexe_usager"].map({1:0,2:1})

    for col in ["localisation_agglo","is_weekend","is_ferie"]:
        df_train[col] = df_train[col].astype(int)
        df_test[col] = df_test[col].astype(int)

    # Target encoding
    target_cols = {
        "equipement_secu_2":"secu2_encoded",
        "equipement_secu_1":"secu_encoded",
        "motif_deplacement":"motif_encoded",
        "obstacle_mobile":"obsm_encoded",
        "point_choc":"choc_encoded",
        "trace_plan":"trace_encoded",
        "obstacle_fixe":"obs_fixe_encoded",
        "manœuvre_avant_accident":"manoeuvre_encoded",
        "type_motorisation":"moteur_encoded",
        "regime_circulation":"regime_encoded",
        "nb_voies":"voies_encoded",
        "profil_route":"profil_encoded",
        "type_collision":"collision_encoded",
        "categorie_route":"route_encoded",
        "intersection":"intersection_encoded",
        "meteo":"meteo_encoded",
        "etat_surface":"surface_encoded",
        "amenagement":"amenagement_encoded",
        "situation_accident":"situation_encoded",
        "voie_reservee":"voie_reservee_encoded",
        "sens_circulation":"sens_circulation_encoded"
    }

    for col_src, col_enc in target_cols.items():
        enc = ce.TargetEncoder(cols=[col_src], smoothing=10)
        enc.fit(df_train[col_src], df_train["gravite"])
        df_train[col_enc] = enc.transform(df_train[col_src])
        df_test[col_enc] = enc.transform(df_test[col_src])

    # One-hot
    df_train = pd.get_dummies(df_train, columns=["categorie_usager","type_vehicule_simplifie"])
    df_test = pd.get_dummies(df_test, columns=["categorie_usager","type_vehicule_simplifie"])
    df_test = df_test.reindex(columns=df_train.columns, fill_value=0)

    # Retrait colonnes inutiles
    df_train = df_train.drop(columns=["TRANCHE_HORAIRE","sexe_usager","type_vehicule"], errors="ignore")
    df_test = df_test.drop(columns=["TRANCHE_HORAIRE","sexe_usager","type_vehicule"], errors="ignore")

    # Sauvegarde
    X_train = df_train.drop(columns=["gravite"])
    y_train = df_train["gravite"]
    X_test = df_test.drop(columns=["gravite"])
    y_test = df_test["gravite"]

    X_train.to_csv(os.path.join(OUTPUT_DIR,"X_train.csv"), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR,"X_test.csv"), index=False)
    y_train.to_csv(os.path.join(OUTPUT_DIR,"y_train.csv"), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DIR,"y_test.csv"), index=False)

if __name__ == "__main__":
    run_encoding_pipeline()
