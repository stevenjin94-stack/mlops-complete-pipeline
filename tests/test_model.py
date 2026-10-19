import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_preprocessing import (
    build_preprocessor,
    split_features_target,
)
from src.evaluate import evaluate_classification


DATA_PATH = "data/raw/employee_attrition.csv"

DROP_COLUMNS = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours",
]


def build_test_model():
    df = pd.read_csv(DATA_PATH)

    # Use a small sample for fast tests.
    df = df.sample(
        n=600,
        random_state=42,
    )

    X, y = split_features_target(
        df,
        "Attrition",
        DROP_COLUMNS,
    )

    y = y.map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )
    )

    preprocessor = build_preprocessor(
        X_train
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    return model, X_test, y_test


def test_predictions_have_correct_shape():
    model, X_test, y_test = (
        build_test_model()
    )

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)
    assert predictions.ndim == 1


def test_model_meets_minimum_performance():
    model, X_test, y_test = (
        build_test_model()
    )

    metrics = evaluate_classification(
        model,
        X_test,
        y_test,
    )

    # A deliberately modest threshold makes
    # this a useful regression test without
    # making CI excessively brittle.
    assert metrics["f1"] >= 0.35
