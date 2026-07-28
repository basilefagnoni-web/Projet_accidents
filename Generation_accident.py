import os
import pandas as pd
import numpy as np

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")

# Charger X_train.zip pour récupérer les colonnes ET les vraies distributions
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.zip"))

SECU_MAPPING = {
    "Ceinture": 2.146,
    "Casque": 3.371,
    "Dispositif enfants": 2.447,
    "Gilet réfléchissant": 2.906,
    "Airbag (2RM/3RM)": 2.726,
    "Gants (2RM/3RM)": 3.089,
    "Gants + Airbag (2RM/3RM)": 2.764,
}
COLLISION_MAPPING = {
    "Frontale": 2.612,
    "Latérale": 2.467,
    "Arrière": 2.495,
    "Multiple": 2.249,
    "Autre": 2.878,
}
MANOEUVRE_MAPPING = {
    "Dépassement": 3.016,
    "Changement de voie": 2.632,
    "Virage": 2.083,
    "Stationnement": 2.088,
    "Autre": 2.504,
}
OBSM_MAPPING = {
    "Véhicule": 2.772,
    "Piéton": 2.402,
    "Animal": 2.849,
    "Autre": 1.650,
}
OBS_FIXE_MAPPING = {
    "Aucun obstacle": 2.688,
    "Véhicule en stationnement": 2.875,
    "Arbre": 2.890,
    "Glissière / séparateur de voie": 3.037,
    "Bâtiment / mobilier urbain": 2.960,
    "Obstacle de bord de chaussée": 2.967,
}

# Colonnes gérées explicitement ailleurs (binaires, catégorielles ordinales,
# one-hot) : on ne veut PAS les tirer depuis X_train pour celles-ci, elles
# ont leur propre logique dédiée plus bas.
COLONNES_BINAIRES = ["localisation_agglo", "luminosite", "is_weekend", "is_ferie", "sexe_encoded"]
COLONNE_HORAIRE = "tranche_horaire_ord"
PREFIXES_ONE_HOT = ("categorie_usager_", "type_vehicule_simplifie_")


def generer_accident(secu, collision, age, manoeuvre, obstacle_mobile, vitesse_max, obs_fixe):
    accident = {
        "secu_encoded": SECU_MAPPING[secu],
        "collision_encoded": COLLISION_MAPPING[collision],
        "age_usager": age,
        "manoeuvre_encoded": MANOEUVRE_MAPPING[manoeuvre],
        "obsm_encoded": OBSM_MAPPING[obstacle_mobile],
        "vitesse_max": vitesse_max,
        "obs_fixe_encoded": OBS_FIXE_MAPPING[obs_fixe],
    }

    # Génération aléatoire pour toutes les autres colonnes
    for col in X_train.columns:
        if col in accident:
            continue
        if col in COLONNES_BINAIRES:
            accident[col] = np.random.choice([0, 1])
        elif col == COLONNE_HORAIRE:
            accident[col] = np.random.choice([1, 2, 3, 4])
        elif col.startswith(PREFIXES_ONE_HOT):
            accident[col] = 0  # sera corrigé juste après (one-hot)
        else:
            # On tire une VRAIE valeur observée dans X_train pour cette colonne,
            # plutôt qu'un intervalle arbitraire — garantit une valeur réaliste
            # et une vraie diversité, quelle que soit l'échelle propre à chaque
            # colonne encodée.
            accident[col] = X_train[col].sample(n=1).values[0]

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
