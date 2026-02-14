import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib


# -----------------------------
# PATH SETUP (PRODUCTION SAFE)
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "salaries.csv")
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")


# -----------------------------
# LOAD DATA
# -----------------------------

def load_data():
    df = pd.read_csv(DATA_PATH)

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df


# -----------------------------
# SPLIT FEATURES & TARGET
# -----------------------------

def split_features_target(df):
    X = df.drop("leaveornot", axis=1)
    y = df["leaveornot"]
    return X, y


# -----------------------------
# BUILD PREPROCESSOR
# -----------------------------

def build_preprocessor():

    categorical_features = [
        "education",
        "city",
        "gender",
        "everbenched"
    ]

    numerical_features = [
        "joiningyear",
        "paymenttier",
        "age",
        "experienceincurrentdomain"
    ]

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", "passthrough", numerical_features)
        ]
    )

    return preprocessor


# -----------------------------
# TRAIN-TEST SPLIT
# -----------------------------

def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y  # important for classification
    )


# -----------------------------
# MAIN EXECUTION
# -----------------------------

if __name__ == "__main__":

    print("Loading data...")
    df = load_data()

    print("Splitting features and target...")
    X, y = split_features_target(df)

    print("Creating train-test split...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Building preprocessing pipeline...")
    preprocessor = build_preprocessor()

    print("Fitting preprocessor...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Ensure backend directory exists
    os.makedirs(BACKEND_DIR, exist_ok=True)

    print("Saving artifacts...")
    joblib.dump(preprocessor, os.path.join(BACKEND_DIR, "preprocessor.pkl"))
    joblib.dump(
        (X_train_processed, X_test_processed, y_train, y_test),
        os.path.join(BACKEND_DIR, "train_test_data.pkl")
    )

    print("✅ Preprocessing completed and artifacts saved successfully.")
