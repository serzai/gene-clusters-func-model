import os
from typing import Tuple, List

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def create_pipeline() -> Pipeline:
    """
    Creates pipeline.
    Uses StandartScaler for numeric features and OrdinalEncoder for categorical.
    """
    numeric_features: List[str] = ["distance", "genes_between", "length_diff"]
    categorical_features: List[str] = ["cog_1", "cog_2", "phylum", "class"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                TargetEncoder(target_type="binary", smooth="auto"),
                categorical_features,
            ),
        ]
    )

    classifier = HistGradientBoostingClassifier(
        random_state=42,
        max_iter=200,
        learning_rate=1e-1,
        early_stopping=True,
        verbose=1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline


def load_and_split_data(
    data_path: str, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads processed dataset and splits into train/test sets.
    """
    df: pd.DataFrame = pd.read_csv(data_path)

    if df.empty:
        raise ValueError("Dataset is empty! Run src/data.py to get processed dataset.")

    X: pd.DataFrame = df.drop("target", axis=1)
    y: pd.Series = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_and_evaluate(data_path: str, model_save_path: str) -> None:
    """
    Main training workflow.
    """
    X_train, X_test, y_train, y_test = load_and_split_data(data_path)

    print(f"Train size: {X_train.shape[0]}")
    print(f"Test size:  {X_test.shape[0]}")

    pipeline: Pipeline = create_pipeline()

    print("Training model")
    pipeline.fit(X_train, y_train)

    print("Evaluating model")
    y_pred = pipeline.predict(X_test)

    accuracy: float = accuracy_score(y_test, y_pred)
    precision: float = precision_score(y_test, y_pred)
    recall: float = recall_score(y_test, y_pred)
    f1: float = f1_score(y_test, y_pred)

    print(f"\n--- Results ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(pipeline, model_save_path)
    print(f"Model saved successfully to {model_save_path}")


if __name__ == "__main__":
    DATA_PATH = "data/processed/pairwise_cogs.csv"
    MODEL_PATH = "models/model_pipeline.joblib"

    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
    else:
        train_and_evaluate(DATA_PATH, MODEL_PATH)
