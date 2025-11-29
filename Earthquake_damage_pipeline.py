import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, mean_squared_error


# ================================
# STEP 1 — LOAD DATASET
# ================================
def load_dataset():
    df = pd.read_csv("Data/Nepal_buildings.csv")

    # Identify damage column (0–5 labels)
    if "damage_grade" not in df.columns:
        raise Exception("Dataset must contain 'damage_grade' column.")

    # Convert damage_grade (0–5) → score 0–10
    df["damage_score"] = (df["damage_grade"] / df["damage_grade"].max()) * 10

    # Convert score → class (low/med/high)
    df["damage_class"] = pd.cut(
        df["damage_score"],
        bins=[-0.1, 3.3, 6.6, 10],
        labels=["low", "medium", "high"]
    )

    # Feature selection
    features = {
        "floors": "count_floors_pre_eq",
        "age": "age",
        "area": "area",
        "material": "foundation_type",
        "roof": "roof_type"
    }

    for k, v in features.items():
        if v not in df.columns:
            raise Exception(f"Column '{v}' not found in dataset.")

    X = df[list(features.values())]
    X.columns = list(features.keys())   # rename nicely

    y_class = df["damage_class"]
    y_score = df["damage_score"]

    return X, y_class, y_score


# ================================
# STEP 2 — PREPROCESSING PIPELINE
# ================================
def build_preprocessor():
    numeric_features = ["floors", "age", "area"]
    categorical_features = ["material", "roof"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    return preprocessor


# ================================
# STEP 3 — TRAIN ALL MODELS
# ================================
def train_all_models():
    X, y_class, y_score = load_dataset()
    pre = build_preprocessor()

    X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
        X, y_class, y_score, test_size=0.2, random_state=42, stratify=y_class
    )

    # Models
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "LinearRegression": LinearRegression()
    }

    trained = {}

    for name, model in models.items():
        print(f"\n=== Training {name} ===")

        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X_train, y_class_train if name != "LinearRegression" else y_score_train)
        trained[name] = pipe

        if name != "LinearRegression":
            print(f"\n{name} Classification Report:")
            print(classification_report(y_class_test, pipe.predict(X_test)))
        else:
            preds = pipe.predict(X_test)
            print("\nLinear Regression MSE:", mean_squared_error(y_score_test, preds))

    # Save all models
    for name, pipe in trained.items():
        joblib.dump(pipe, f"models/{name}.pkl")

    print("\nMODELS SAVED IN /models FOLDER!")

    return trained


# ================================
# STEP 4 — PREDICTION FUNCTION
# ================================
def predict_damage():
    rf = joblib.load("models/RandomForest.pkl")

    sample = pd.DataFrame([{
        "floors": 3,
        "age": 40,
        "area": 120,
        "material": "Mud mortar-Stone/Brick",
        "roof": "Bamboo/Timber-Light roof"
    }])

    p = rf.predict(sample)[0]
    print("\nPredicted damage class:", p)


# ================================
# MAIN
# ================================
if __name__ == "__main__":
    train_all_models()
    predict_damage()
