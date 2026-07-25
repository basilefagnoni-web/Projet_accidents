import os
import pandas as pd
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
import joblib

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_r6.pkl")

def binariser(y):
    y = y.values.ravel()
    return np.where(np.isin(y, [1,4]), 0, 1)

def train_model():
    X_train = pd.read_csv(os.path.join(DATA_DIR,"X_train.zip"))
    y_train = pd.read_csv(os.path.join(DATA_DIR,"y_train.zip"))
    y_train_bin = binariser(y_train)

    model = BalancedRandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        sampling_strategy=1.0,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train_bin)
    os.makedirs(os.path.join(BASE_DIR,"models"), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

def load_model():
    return joblib.load(MODEL_PATH)

def load_columns():
    return pd.read_csv(os.path.join(DATA_DIR,"X_train.zip")).columns

def predict_gravity(inputs: dict):
    model = load_model()
    cols = load_columns()
    df = pd.DataFrame([inputs])[cols]
    pred = model.predict(df)[0]
    return int(pred)

if __name__ == "__main__":
    train_model()
