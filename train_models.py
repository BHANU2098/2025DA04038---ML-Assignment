"""
Train all 5 classification models on the Adult Income dataset and save them.
Run this script once to generate saved model files and test_data.csv.

Usage:
    python model/train_models.py
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ── 1. Load dataset ──────────────────────────────────────────────────────────
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

print("Downloading Adult Income dataset...")
df = pd.read_csv(URL, names=COLUMNS, na_values=" ?", skipinitialspace=True)
df.dropna(inplace=True)
print(f"Dataset shape: {df.shape}")

# ── 2. Preprocess ─────────────────────────────────────────────────────────────
le = LabelEncoder()
cat_cols = df.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop("income", axis=1)
y = df["income"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Save test data for Streamlit upload
test_df = X_test.copy()
test_df["income"] = y_test.values
test_df.to_csv("test_data.csv", index=False)
print("Saved test_data.csv")

# ── 3. Train models ───────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":        DecisionTreeClassifier(random_state=42),
    "kNN":                  KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes":          GaussianNB(),
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
}

os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.pkl")

results = []

for name, model in models.items():
    # kNN and Naive Bayes benefit from scaling; tree-based models don't require it
    if name in ("Logistic Regression", "kNN", "Naive Bayes"):
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "AUC":       round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall":    round(recall_score(y_test, y_pred), 4),
        "F1":        round(f1_score(y_test, y_pred), 4),
        "MCC":       round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)

    safe_name = name.replace(" ", "_")
    joblib.dump(model, f"model/{safe_name}.pkl")
    print(f"\n{name}")
    print(f"  Accuracy : {metrics['Accuracy']}")
    print(f"  AUC      : {metrics['AUC']}")
    print(f"  Precision: {metrics['Precision']}")
    print(f"  Recall   : {metrics['Recall']}")
    print(f"  F1       : {metrics['F1']}")
    print(f"  MCC      : {metrics['MCC']}")
    print(classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))

print("\n=== Comparison Table ===")
results_df = pd.DataFrame(results).set_index("Model")
print(results_df.to_string())
