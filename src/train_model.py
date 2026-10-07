"""Preprocess the real Telco churn data, then train and compare models."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

from src.load_data import load

TARGET = "Churn"
ID_COLS = ["customerID"]

df = load()
print(f"Loaded {len(df)} customers")
print(f"Churn rate: {df[TARGET].value_counts(normalize=True).get('Yes', 0):.1%}")

df = df.drop(columns=[c for c in ID_COLS if c in df.columns])

# TotalCharges arrives as text with blanks for brand-new customers.
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

y = (df[TARGET] == "Yes").astype(int)
X = df.drop(columns=[TARGET])

numeric = X.select_dtypes(include=["number"]).columns.tolist()
categorical = X.select_dtypes(exclude=["number"]).columns.tolist()

preprocess = ColumnTransformer(
    [
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", __import__("sklearn").preprocessing.OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=300, random_state=42, eval_metric="logloss",
        use_label_encoder=False, tree_method="hist",
    ),
}

baseline = max(y_test.mean(), 1 - y_test.mean())
print(f"\nBaseline (always predict majority): {baseline:.3f}")
print("=" * 72)

results = {}
for name, model in models.items():
    pipe = Pipeline([("pre", preprocess), ("clf", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    results[name] = (pipe, auc)
    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"AUC      : {auc:.3f}")
    if name == "RandomForest":
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion matrix:")
        print(cm)

best_name = max(results, key=lambda k: results[k][1])
print(f"\nBest model by AUC: {best_name} ({results[best_name][1]:.3f})")

pipe = results[best_name][0]
y_prob = pipe.predict_proba(X_test)[:, 1]
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_prob, alpha=0.3, s=10)
plt.xlabel("Actual churn (0/1)")
plt.ylabel("Predicted probability")
plt.title(f"Churn probability separation ({best_name})")
plt.tight_layout()
plt.savefig("plots_churn.png", dpi=120)
print("Saved plot to plots_churn.png")
