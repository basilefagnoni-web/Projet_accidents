import streamlit as st
import pandas as pd

st.title("Comparaison des modèles — Classification à 4 classes")

st.write(
    "Avant de passer à une approche binaire en cascade, on a d'abord testé "
    "la prédiction directe sur les 4 classes de gravité : Indemne, Tué, "
    "Hospitalisé, Blessé léger."
)

st.subheader("Matrices de confusion des 5 modèles testés")
st.image("confusion_matrices_4classes.png")

st.subheader("Tableau comparatif des recalls")
synthese_4c = pd.read_csv("synthese_4classes.csv")
st.dataframe(
    synthese_4c.style.background_gradient(subset=["Recall Tué"], cmap="Reds")
)

st.markdown(
    """
**Pourquoi se concentrer sur le recall de la classe "Tué" ?**

Dans ce projet, l'enjeu n'est pas de bien classer *tous* les cas de façon
égale : une erreur qui manque un décès potentiel (faux négatif sur "Tué")
est bien plus coûteuse qu'une erreur qui classe un accident bénin comme
plus grave qu'il ne l'est. Le recall mesure justement la capacité du
modèle à ne pas laisser passer les vrais cas de la classe qui nous
intéresse — ici, "Tué". C'est donc la métrique la plus alignée avec
l'objectif métier du projet, plus que l'accuracy globale, qui peut rester
élevée même en ratant presque tous les cas de la classe la plus rare.
"""
)

st.markdown(
    """
**Pourquoi passer à une approche binaire ?**

Sur les 4 classes, la classe "Tué" est à la fois la plus rare et la plus
critique à détecter — et c'est précisément celle sur laquelle tous les
modèles testés obtiennent les moins bons résultats. En regroupant d'abord
les classes en "Grave" / "Non grave", puis en ne distinguant "Hospitalisé"
de "Tué" que parmi les cas déjà identifiés comme graves, chaque modèle a
une tâche plus simple et plus ciblée à résoudre — avec des gains mesurables
sur le recall, comme le montre la page "Comparaison des modèles — Approche
binaire".
"""
)
