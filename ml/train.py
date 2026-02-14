import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.model_selection import GridSearchCV


# -----------------------------
# PATH SETUP
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")

DATA_FILE = os.path.join(BACKEND_DIR, "train_test_data.pkl")


# -----------------------------
# LOAD DATA
# -----------------------------

def load_data():
    return joblib.load(DATA_FILE)


# -----------------------------
# EVALUATION FUNCTION
# -----------------------------

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


# -----------------------------
# MAIN EXECUTION
# -----------------------------

if __name__ == "__main__":

    print("Loading processed data...")
    X_train, X_test, y_train, y_test = load_data()

    # =========================================================
    # 1️⃣ BASELINE MODEL — Logistic Regression
    # =========================================================
    print("\nTraining Logistic Regression (baseline)...")

    log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    log_model.fit(X_train, y_train)

    evaluate_model("Logistic Regression", log_model, X_test, y_test)

    # =========================================================
    # 2️⃣ TUNED RANDOM FOREST (GridSearch)
    # =========================================================
    print("\nRunning GridSearch for Random Forest...")

    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    rf = RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    )

    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_

    print("\nBest Parameters:", grid_search.best_params_)

    evaluate_model("Tuned Random Forest (Threshold=0.5)", best_rf, X_test, y_test)

    # =========================================================
    # 3️⃣ THRESHOLD TUNING
    # =========================================================
    print("\n--- Threshold Tuning ---")

    y_proba = best_rf.predict_proba(X_test)[:, 1]

    # Default threshold
    default_threshold = 0.5

    # Business-optimized threshold
    custom_threshold = 0.4

    y_pred_custom = (y_proba >= custom_threshold).astype(int)

    print(f"\nCustom Threshold = {custom_threshold}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_custom))
    print("\nClassification Report:\n",
          classification_report(y_test, y_pred_custom))

    # =========================================================
    # 4️⃣ SAVE FINAL MODEL & THRESHOLD
    # =========================================================
    os.makedirs(BACKEND_DIR, exist_ok=True)

    joblib.dump(best_rf, os.path.join(BACKEND_DIR, "model.pkl"))
    joblib.dump(custom_threshold, os.path.join(BACKEND_DIR, "threshold.pkl"))

    print("\n✅ Final model and threshold saved successfully.")
