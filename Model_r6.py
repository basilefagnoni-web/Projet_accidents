import os
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")

# Chargement des données ZIP
x_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.zip"))
x_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.zip"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.zip"))
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.zip"))

# Nettoyage
def nettoyer_cible(y):
    col = y.iloc[:,0].astype(str).str.strip().replace("", np.nan)
    return col.astype(float).astype("Int64")

y_train_clean = nettoyer_cible(y_train)
y_test_clean  = nettoyer_cible(y_test)

# Binarisation grave / non grave
def binariser(y):
    y = y.values.ravel()
    return np.where(np.isin(y, [1, 4]), 0, 1)

y_train_bin = binariser(y_train_clean)
y_test_bin  = binariser(y_test_clean)

# Modèle r6
modele_r6 = BalancedRandomForestClassifier(
    n_estimators=80,        # au lieu de 300
    max_depth=12,          # au lieu de 20
    min_samples_split=4,
    sampling_strategy=1.0,
    random_state=42,
    n_jobs=-1
)

modele_r6.fit(x_train, y_train_bin)

# Modèle hospitalisé / tué
mask_train_graves = y_train_clean.isin([2,3])
mask_test_graves  = y_test_clean.isin([2,3])

train_graves = x_train.loc[mask_train_graves]
test_graves  = x_test.loc[mask_test_graves]

y_train_graves = y_train_clean.loc[mask_train_graves].map({2:1, 3:0})
y_test_graves  = y_test_clean.loc[mask_test_graves].map({2:1, 3:0})

pipeline_grave = Pipeline([
    ("smote", SMOTE(sampling_strategy=0.5, random_state=42)),
    ("rf", RandomForestClassifier(
        n_estimators=80,    # au lieu de 300
        max_depth=12,
        class_weight={0:1, 1:10},
        random_state=42,
        n_jobs=-1
    ))
])

pipeline_grave.fit(train_graves, y_train_graves)

# Sauvegarde des modèles
os.makedirs("models", exist_ok=True)
joblib.dump(modele_r6, "models/modele_r6.pkl")
joblib.dump(pipeline_grave, "models/pipeline_grave.pkl")

print("Modèles sauvegardés dans /models/")
