import streamlit as st
import pandas as pd

st.title(" Analyse des données d'accidents")

df = pd.read_csv("data/accidents.csv")

st.subheader("Aperçu des données")
st.dataframe(df.head())

st.subheader("Statistiques générales")
st.write(df.describe())

st.subheader("Distribution de la gravité")
st.bar_chart(df["gravite"].value_counts())
