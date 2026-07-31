import streamlit as st

st.set_page_config(page_title="Projet Accidents", layout="wide")

exploration = st.Page(
    "pages/Resume_rapport_exploration.py",
    title="Exploration",
    icon="🔍"
)
comparaison_4c = st.Page(
    "pages/Comparaison_modeles_4classes.py",
    title="Comparaison modèles - 4 classes",
    icon="📊"
)
comparaison_bin = st.Page(
    "pages/Comparaison_modeles_binaire.py",
    title="Comparaison modèles - binaire",
    icon="📈"
)
test_gravite = st.Page(
    "pages/Test_gravite.py",
    title="Test gravité",
    icon="🚗"
)

pg = st.navigation([exploration, comparaison_4c, comparaison_bin, test_gravite])
pg.run()
