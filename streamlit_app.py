import streamlit as st
import pandas as pd
from Code_complet import load_columns, predict_gravity

st.title(" Prédiction de la gravité d’un accident de la route")

cols = load_columns()
inputs = {}

st.header(" Décris un accident")

for col in cols:
    if "encoded" in col or col.startswith(("secu","obs","choc","meteo","route")):
        inputs[col] = st.slider(col, 0.0, 5.0, 2.5)
    elif col.startswith(("is_","categorie_usager","type_vehicule_simplifie")):
        inputs[col] = st.selectbox(col, [0,1])
    else:
        inputs[col] = st.number_input(col, value=0.0)

if st.button(" Prédire"):
    pred = predict_gravity(inputs)
    label = "Grave" if pred == 1 else "Non grave"
    st.metric("Gravité prédite", label)
