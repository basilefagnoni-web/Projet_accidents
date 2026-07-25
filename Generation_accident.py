import os
import pandas as pd
import numpy as np

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")

# Charger X_train.zip pour récupérer les colonnes
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.zip"))

SECU_MAPPING = {
    "Ceinture": 2.2,
    "Casque": 2.7,
    "Dispositif enfants": 2.9,
    "Gilet réfléchissant": 2.4,
    "Airbag (2RM/3RM)": 2.8,
    "Gants (2RM/3RM)": 2.5,
    "Gants + Airbag (2RM/3RM)": 2.9
}

COLLISION_MAPPING = {
    "Frontale": 2.7,
    "Latérale": 2.5,
    "Arrière": 2.3,
    "Multiple": 2.8,
    "Autre": 2.4
}

MANOEUVRE_MAPPING = {
    "Dépassement": 2.6,
    "Changement de voie": 2.5,
    "Virage": 2.4,
    "Freinage": 2.3,
    "Stationnement": 2.1,
    "Autre": 2.2
}

OBSM_MAPPING = {
    "Véhicule": 2.5,
    "Piéton": 2.7,
    "Cycliste": 2.6,
    "Animal": 2.3,
    "Objet": 2.2,
    "Autre": 2.4
}

def generer_accident(secu, collision, age, manoeuvre, obstacle_mobile):

    accident = {
        "secu_encoded": SECU_MAPPING[secu],
        "collision_encoded": COLLISION_MAPPING[collision],
        "age_usager": age,
        "manoeuvre_encoded": MANOEUVRE_MAPPING[manoeuvre],
        "obsm_encoded": OBSM_MAPPING[obstacle_mobile],
    }

    # Génération aléatoire pour toutes les autres colonnes
    for col in X_train.columns:
        if col not in accident:

            if col in ["localisation_agglo", "luminosite", "is_weekend", "is_ferie", "sexe_encoded"]:
                accident[col] = np.random.choice([0, 1])

            elif col == "tranche_horaire_ord":
                accident[col] = np.random.choice([1, 2, 3, 4])

            elif col.startswith("categorie_usager_") or col.startswith("type_vehicule_simplifie_"):
                accident[col] = 0

            else:
                accident[col] = np.random.uniform(2.2, 2.8)

    # One-hot corrections
    cats = ["categorie_usager_1", "categorie_usager_2", "categorie_usager_3"]
    chosen = np.random.choice(cats)
    for c in cats:
        accident[c] = 1 if c == chosen else 0

    vehs = [
        "type_vehicule_simplifie_Voiture",
        "type_vehicule_simplifie_Deux_Roues_Moteur",
        "type_vehicule_simplifie_Velo_Trotinette",
        "type_vehicule_simplifie_Poids_Lourd_Utilitaire",
        "type_vehicule_simplifie_Transport_Commun",
        "type_vehicule_simplifie_Autre"
    ]
    chosen_v = np.random.choice(vehs)
    for v in vehs:
        accident[v] = 1 if v == chosen_v else 0

    df_accident = pd.DataFrame([accident])
    df_accident = df_accident[X_train.columns]

    return df_accident
