
# train.py
# ML Zoomcamp 2026 Midterm Project
# Reproducible training for 4G/5G signal level prediction

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBRegressor


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "xgboost_signal_model.bin"

FEATURES = [
    "Latitude",
    "Longitude",
    "Speed",
    "Operatorname",
    "NetworkTech",
    "Mobility",
    "Altitude",
]

CATEGORICAL = [
    "Operatorname",
    "NetworkTech",
    "Mobility",
]

NUMERICAL = [
    "Latitude",
    "Longitude",
    "Speed",
    "Altitude",
]


def load_data():
    files = list(DATA_DIR.rglob("processed_data_cleaned.v2.csv"))

    if len(files) != 1:
        raise FileNotFoundError(
            f"Expected one dataset file, found {len(files)}"
        )

    df = pd.read_csv(files[0])
    print(f"Loaded dataset: {df.shape}")
    return df


def split_data(df):
    X = df[FEATURES].copy()
    y = df["Level"].copy()
    groups = df["SessionID"]

    split1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.40,
        random_state=42,
    )

    train_idx, temp_idx = next(
        split1.split(X, y, groups=groups)
    )

    split2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=42,
    )

    val_relative_idx, test_relative_idx = next(
        split2.split(
            X.iloc[temp_idx],
            y.iloc[temp_idx],
            groups=groups.iloc[temp_idx],
        )
    )

    val_idx = temp_idx[val_relative_idx]
    test_idx = temp_idx[test_relative_idx]

    train_sessions = set(groups.iloc[train_idx])
    val_sessions = set(groups.iloc[val_idx])
    test_sessions = set(groups.iloc[test_idx])

    assert train_sessions.isdisjoint(val_sessions)
    assert train_sessions.isdisjoint(test_sessions)
    assert val_sessions.isdisjoint(test_sessions)

    X_train_full = pd.concat(
        [X.iloc[train_idx], X.iloc[val_idx]]
    )
    y_train_full = pd.concat(
        [y.iloc[train_idx], y.iloc[val_idx]]
    )

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    print(f"Training records: {len(X_train_full)}")
    print(f"Test records: {len(X_test)}")

    return X_train_full, X_test, y_train_full, y_test


def build_model():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL,
            ),
        ]
    )

    model = XGBRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def main():
    print("ML Zoomcamp Midterm - XGBoost Training")
    print("--------------------------------------")

    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df)

    model = build_model()

    print("\nTraining final model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )
    mae = mean_absolute_error(y_test, predictions)

    print("\nFinal Test Results")
    print("------------------")
    print(f"Test RMSE: {rmse:.4f} dB")
    print(f"Test MAE:  {mae:.4f} dB")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"\nModel saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
