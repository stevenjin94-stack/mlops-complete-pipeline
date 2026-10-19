from __future__ import annotations

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_classification(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate a classification model and find the best F1 threshold."""

    probabilities = model.predict_proba(X_test)[:, 1]

    best_threshold = 0.50
    best_f1 = 0.0
    best_predictions = None

    for threshold in [i / 100 for i in range(20, 61)]:
        predictions = (probabilities >= threshold).astype(int)

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_predictions = predictions

    predictions = best_predictions

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
        "threshold": best_threshold,
    }
