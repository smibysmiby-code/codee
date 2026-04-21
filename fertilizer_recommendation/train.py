"""Fertilizer recommendation — corrected training pipeline.

Fixes the common 14%-accuracy (random-guessing on 7 classes) bug by:
  - fitting all encoders/scalers inside a single Pipeline
  - using StratifiedKFold on a tiny dataset instead of one split
  - keeping features and labels aligned at all times
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).parent / "Fertilizer Prediction.csv"
MODEL_PATH = Path(__file__).parent / "fertilizer_model.joblib"
RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def build_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def sanity_checks(df: pd.DataFrame, target: str) -> None:
    print("Shape:", df.shape)
    print("Null counts:\n", df.isnull().sum())
    print("\nTarget distribution:\n", df[target].value_counts())
    print("\nDtypes:\n", df.dtypes)


def main() -> None:
    target = "Fertilizer Name"
    df = load_data(DATA_PATH)
    sanity_checks(df, target)

    X = df.drop(columns=[target])
    y_raw = df[target]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    print(f"\nNumeric features: {numeric_cols}")
    print(f"Categorical features: {categorical_cols}")

    pipeline = build_pipeline(numeric_cols, categorical_cols)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"\nBaseline RF 5-fold CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    param_grid = {
        "clf__n_estimators": [200, 400, 600],
        "clf__max_depth": [None, 8, 16],
        "clf__min_samples_split": [2, 4],
    }
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    search.fit(X_train, y_train)

    print(f"\nBest CV accuracy: {search.best_score_:.3f}")
    print(f"Best params: {search.best_params_}")

    y_pred = search.predict(X_test)
    print("\nHold-out classification report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(
        {"model": search.best_estimator_, "label_encoder": label_encoder},
        MODEL_PATH,
    )
    print(f"\nSaved model to {MODEL_PATH}")


def predict(sample: dict) -> str:
    """Load the saved bundle and predict a fertilizer for one sample."""
    bundle = joblib.load(MODEL_PATH)
    model: Pipeline = bundle["model"]
    label_encoder: LabelEncoder = bundle["label_encoder"]
    X = pd.DataFrame([sample])
    pred = model.predict(X)
    return label_encoder.inverse_transform(pred)[0]


if __name__ == "__main__":
    main()
