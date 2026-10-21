from __future__ import annotations

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

def select_classification_threshold(
    model,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> float:
    """Select the threshold that maximizes F1 on validation data."""

    probabilities = model.predict_proba(X_validation)[:, 1]

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in [i / 100 for i in range(20, 61)]:
        predictions = (probabilities >= threshold).astype(int)

        score = f1_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold


def evaluate_classification(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.50,
) -> dict[str, float]:
    """Evaluate test data using a previously selected threshold."""

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "threshold": threshold,
    }
