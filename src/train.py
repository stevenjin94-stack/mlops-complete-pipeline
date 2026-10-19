from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn import metrics
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_preprocessing import (
    build_preprocessor,
    split_features_target,
)
from src.evaluate import evaluate_classification


def load_config(path: str) -> dict:
    """Load YAML configuration."""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_file_hash(path: str) -> str:
    """Calculate a SHA256 hash of the dataset."""
    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            sha256.update(block)

    return sha256.hexdigest()


def build_model(config: dict, overrides: dict | None = None):
    """Build model from YAML configuration."""
    model_config = config["model"].copy()

    if overrides:
        model_config.update(
            {
                key: value
                for key, value in overrides.items()
                if value is not None
            }
        )

    if model_config["type"] != "random_forest":
        raise ValueError(
            f"Unsupported model type: {model_config['type']}"
        )

    return RandomForestClassifier(
        n_estimators=model_config["n_estimators"],
        max_depth=model_config["max_depth"],
        min_samples_split=model_config["min_samples_split"],
        min_samples_leaf=model_config["min_samples_leaf"],
        class_weight=model_config["class_weight"],
        random_state=config["project"]["random_state"],
        n_jobs=-1,
    )


def train(
    config_path: str = "configs/config.yaml",
    overrides: dict | None = None,
) -> dict:
    """Train, evaluate, and log one MLflow experiment."""

    config = load_config(config_path)

    data_path = config["data"]["raw_path"]
    target_column = config["data"]["target_column"]
    drop_columns = config["features"]["drop_columns"]

    df = pd.read_csv(data_path)

    X, y = split_features_target(
        df,
        target_column,
        drop_columns,
    )

    y = y.map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Unexpected target values detected.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["project"]["random_state"],
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)
    classifier = build_model(config, overrides)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    mlflow.set_tracking_uri(
        config["training"]["mlflow_tracking_uri"]
    )

    mlflow.set_experiment(
        config["training"]["experiment_name"]
    )

    with mlflow.start_run() as run:
        model_config = config["model"].copy()

        if overrides:
            model_config.update(
                {
                    key: value
                    for key, value in overrides.items()
                    if value is not None
                }
            )

        mlflow.log_params(
            {
                "model_type": model_config["type"],
                "n_estimators": model_config["n_estimators"],
                "max_depth": model_config["max_depth"],
                "min_samples_split": model_config[
                    "min_samples_split"
                ],
                "min_samples_leaf": model_config[
                    "min_samples_leaf"
                ],
                "class_weight": model_config["class_weight"],
                "test_size": config["data"]["test_size"],
                "random_state": config["project"]["random_state"],
            }
        )

        data_hash = get_file_hash(data_path)

        mlflow.log_param(
            "data_version",
            data_hash,
        )

        pipeline.fit(X_train, y_train)

        metrics = evaluate_classification(
            pipeline,
            X_test,
            y_test,
        )

        mlflow.log_metrics(
            {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )

        mlflow.log_param(
            "classification_threshold",
            metrics["threshold"],
        )

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
            ],
        )

        print(f"Run ID: {run.info.run_id}")
        print("Metrics:")

        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

        minimum_f1 = config["training"]["minimum_f1"]

        if metrics["f1"] < minimum_f1:
            raise RuntimeError(
                f"Model F1={metrics['f1']:.4f} is below "
                f"minimum threshold {minimum_f1:.4f}"
            )



    return {
        "run_id": run.info.run_id,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
    )

    args = parser.parse_args()

    train(args.config)


if __name__ == "__main__":
    main()
