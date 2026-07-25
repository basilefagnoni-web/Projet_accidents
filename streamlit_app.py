import streamlit as st
from generator import generer_accident
from predict import predict_gravite

st.title("Prédiction de gravité d'accident")

secu = st.selectbox("Sécurité", list(SECU_MAPPING.keys()))
collision = st.selectbox("Collision", list(COLLISION_MAPPING.keys()))
manoeuvre = st.selectbox("Manœuvre", list(MANOEUVRE_MAPPING.keys()))
obstacle_mobile = st.selectbox("Obstacle mobile", list(OBSM_MAPPING.keys()))
age = st.slider("Âge", 14, 99, 40)

if st.button("Prédire"):
    accident = generer_accident(secu, collision, age, manoeuvre, obstacle_mobile)
    result = predict_gravite(accident)

    st.write("Gravité :", result["gravite_binaire"][0])
    st.write("Détail :", result["gravite_detaillee"][0])
