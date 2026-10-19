from __future__ import annotations

from typing import Iterable

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_DROP_COLUMNS = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours",
]


def validate_dataframe(
    df: pd.DataFrame,
    target_column: str,
) -> None:
    """Validate the input dataframe."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("DataFrame cannot be empty.")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found."
        )


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
    drop_columns: Iterable[str] | None = None,
):
    """Separate features and target without modifying the original dataframe."""
    validate_dataframe(df, target_column)

    drop_columns = list(drop_columns or [])

    working = df.copy(deep=True)

    columns_to_drop = [
        column
        for column in drop_columns
        if column in working.columns
    ]

    X = working.drop(
        columns=[target_column] + columns_to_drop
    )

    y = working[target_column].copy()

    return X, y


def identify_feature_types(
    X: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Identify numeric and categorical columns."""
    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return numeric_columns, categorical_columns


def build_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:
    """
    Build an sklearn preprocessing pipeline.

    Numeric values use median imputation.
    Categorical values use most-frequent imputation and one-hot encoding.
    """
    numeric_columns, categorical_columns = identify_feature_types(X)

    if not numeric_columns and not categorical_columns:
        raise ValueError("No usable feature columns were found.")

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


def preprocess_dataframe(
    df: pd.DataFrame,
    target_column: str,
    drop_columns: Iterable[str] | None = None,
):
    """
    Return X, y, and a fitted preprocessing transformer.

    This function does not modify df.
    """
    X, y = split_features_target(
        df,
        target_column,
        drop_columns,
    )

    preprocessor = build_preprocessor(X)

    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor
