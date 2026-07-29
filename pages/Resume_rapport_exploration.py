import streamlit as st

st.title("Rapport d'exploration et de pre-processing")
st.caption("Prédiction de la gravité des accidents routiers en France — Bases BAAC 2019-2024")

# =============================================================
# 1. Contexte et objectif
# =============================================================

st.header("1. Contexte et objectif")
st.markdown(
    "Prédire la gravité des accidents corporels de la circulation en France "
    "à partir des bases **BAAC** (Bulletin d'Analyse des Accidents Corporels), "
    "sur la période **2019-2024** (6 ans d'historique)."
)

# =============================================================
# 2. Les 4 jeux de données sources
# =============================================================

st.header("2. Quatre jeux de données sources")
st.markdown(
    """
| Jeu de données | Granularité | Contenu principal |
|---|---|---|
| Caractéristiques | 1 ligne / accident | Date, heure, GPS, luminosité, météo, intersection, collision |
| Lieux | 1 ligne / accident | Type de route, vitesse max autorisée, état de surface |
| Véhicules | 1 ligne / véhicule impliqué | Catégorie, obstacle heurté, manœuvre, motorisation |
| Usagers | 1 ligne / usager impliqué | Place, âge, sexe, équipements de sécurité, **gravité (cible)** |
"""
)

# =============================================================
# 3. Démarche générale
# =============================================================

st.header("3. Démarche de préparation des données")
st.markdown(
    """
1. Compilation des 6 millésimes et nettoyage propre à chaque jeu de données
2. Jointure des 4 tables et contrôle de la volumétrie finale
3. Feature engineering : création, suppression, renommage de variables
4. Encodage des variables catégorielles selon leur nature
5. Standardisation des variables numériques et séparation train / test
"""
)

# =============================================================
# 4. Nettoyage par jeu de données
# =============================================================

st.header("4. Compilation et nettoyage par jeu de données")

with st.expander("4.1 Caractéristiques"):
    st.markdown(
        "- Harmonisation de l'identifiant (Num_Acc / Accident_Id selon les millésimes)\n"
        "- Création de `date_time` à partir de jour/mois/an/hrmn\n"
        "- Suppression de la colonne GPS brute (remplacée par lat/long)\n"
        "- Jeu de données globalement de bonne qualité, peu de valeurs manquantes"
    )

with st.expander("4.2 Lieux"):
    st.markdown(
        "- Suppression des doublons `Num_Acc` (conservation de la ligne la plus complète)\n"
        "- Création de `vma_clean` : correction d'environ une centaine de lignes "
        "avec une vitesse max > 130 km/h (division par 10)"
    )

with st.expander("4.3 Véhicules"):
    st.markdown(
        "- Vérification de l'unicité de la clé (Num_Acc + id_vehicule)\n"
        "- `occutc` : les NA hors bus/autocar sont une absence structurelle, pas une donnée manquante\n"
        "- Vérification de l'absence des codes véhicule obsolètes (période 2019-2024)\n"
        "- Harmonisation des codes -1 (non renseigné) en NA sur l'ensemble des variables concernées"
    )

with st.expander("4.4 Usagers"):
    st.markdown(
        "- Création d'un identifiant unique `id_usager` (environ 30 % des lignes concernées)\n"
        "- `locp`, `actp`, `etatp` : absence structurelle pour les non-piétons, pas un défaut de qualité"
    )

# =============================================================
# 5. Jointures et volumétrie
# =============================================================

st.header("5. Jointures et volumétrie finale")
st.markdown(
    """
| Étape | Table jointe | Clé | Type |
|---|---|---|---|
| 1 | Caractéristiques + Lieux | Num_Acc | Inner |
| 2 | + Véhicules | Num_Acc | Inner |
| 3 | + Usagers | Num_Acc + id_vehicule | Inner |
"""
)
st.metric("Volumétrie finale", "745 158 lignes", help="Granularité : 1 ligne par usager impliqué")
st.caption(
    "Point de vigilance : un même accident et un même véhicule peuvent être associés à "
    "plusieurs usagers (piétons distincts percutés simultanément)."
)

# =============================================================
# 6. Feature engineering
# =============================================================

st.header("6. Feature engineering")

with st.expander("6.1 Variables créées"):
    st.markdown(
        "- `age_usager` = année − année de naissance\n"
        "- `date_time` → décomposé en `jour`, `mois`, `is_weekend`\n"
        "- `tranche_horaire`, construite à partir de `hrmn`"
    )

with st.expander("6.2 - 6.3 Variables supprimées"):
    st.markdown(
        """
**Redondantes ou fortement corrélées** : `place` (corrélée à catégorie usager), `etatp`, `locp`

**Non exploitables** :
- Identifiants techniques (Num_Acc, id_vehicule, id_usager...)
- Données géographiques redondantes (adr, voie, com, dep...) — jusqu'à 92 % de NA selon les colonnes
- `vma` brute (remplacée par `vma_clean`)
- Trop de valeurs manquantes : `occutc` (>90 %), `larrout` (21 %), `secu3` (>90 %), `actp`
- Redondance temporelle : jour, mois, date_time, hrmn, an
"""
    )

with st.expander("6.4 Traitement des valeurs manquantes"):
    st.markdown(
        """
1. Suppression des lignes où la gravité (variable cible) est non renseignée
2. Codes -1 restants → convertis en NA
3. Suppression des lignes avec NA sur les variables critiques (collision, luminosité, météo)
4. `equipement_secu_2` : NA → 0 (aucun équipement)
5. `regime_circulation`, `motif_deplacement`, `sexe_usager` : NA → mode de la variable
6. Variables à faible taux de NA résiduel : suppression des lignes concernées
"""
    )

with st.expander("6.5 Renommage des variables"):
    st.markdown(
        "L'ensemble des variables a été renommé par domaine pour améliorer la lisibilité "
        "(ex : `atm` → `meteo`, `col` → `type_collision`, `catv` → `type_vehicule`, "
        "`grav` → `gravite`, `secu1` → `equipement_secu_1`)."
    )

# =============================================================
# 7. Encodage des variables
# =============================================================

st.header("7. Encodage des variables")
st.markdown(
    """
| Méthode | Variables concernées |
|---|---|
| Label Encoding (binaire) | `localisation_agglo`, `sexe_usager`, `is_weekend`, `is_ferie` |
| Target Encoding | `type_collision`, `categorie_route`, `regime_circulation`, `obstacle_fixe`, `obstacle_mobile`, `manoeuvre_avant_accident`, `equipement_secu_1/2`, etc. |
| One-Hot Encoding | `meteo`, `etat_surface`, `amenagement`, `situation_accident`, `categorie_usager`, etc. |
| Encodage ordinal | `tranche_horaire` |
| Non encodées | `age_usager`, `vitesse_max` (numériques continues) |
"""
)
st.caption(
    "`type_vehicule` (31 catégories d'origine) regroupée en 5 catégories logiques "
    "(`type_vehicule_simplifie`) pour limiter le sur-ajustement."
)

# =============================================================
# 8. Standardisation
# =============================================================

st.header("8. Standardisation et préparation à la modélisation")
st.markdown(
    """
- Séparation X (variables explicatives) / y (`gravite`, variable cible)
- Split train / test réalisé **avant** toute transformation, pour éviter les fuites d'information
- Standardisation appliquée uniquement aux variables numériques non encodées :
  `nb_voies`, `vitesse_max`, `age_usager`
- Le `fit` est réalisé uniquement sur le train, puis appliqué (`transform`) au test
"""
)

# =============================================================
# 9. Synthèse
# =============================================================

st.header("9. Synthèse")
st.markdown(
    "Le pipeline de préparation couvre l'ensemble des étapes nécessaires à la modélisation : "
    "compilation des six millésimes, nettoyage ciblé par jeu de données, jointures contrôlées, "
    "feature engineering documenté et stratégie d'encodage adaptée à la nature de chaque variable."
)
