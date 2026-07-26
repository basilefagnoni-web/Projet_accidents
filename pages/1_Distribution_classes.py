import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def charger_donnees():
    return pd.read_csv("data/accidents_France_encoded.zip")

df = charger_donnees()

st.title(" Distribution des classes")

mapping = {1: "Indemne", 2: "Tué", 3: "Hosp.", 4: "Léger"}
def binariser(y):
    return np.where(np.isin(y, [1, 4]), 0, 1)

y = df["gravite"]
y_bin = binariser(y)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Multi-classes
counts_multi = y.value_counts().sort_index()
sns.barplot(
    x=[mapping[int(i)] for i in counts_multi.index],
    y=counts_multi.values,
    palette="viridis",
    ax=axes[0]
)
axes[0].set_title("Distribution originale — 4 classes")

# Binaire
counts_bin = pd.Series(y_bin).value_counts().sort_index()
sns.barplot(
    x=["Non grave", "Grave"],
    y=counts_bin.values,
    palette=["#5DCAA5", "#D85A30"],
    ax=axes[1]
)
axes[1].set_title("Distribution binaire — Grave / Non grave")

plt.tight_layout()
st.pyplot(fig)
