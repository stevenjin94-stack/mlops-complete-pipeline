from pathlib import Path

import pandas as pd


DATA_PATH = Path(
    "data/raw/employee_attrition.csv"
)

TARGET = "Attrition"


def load_data():
    assert DATA_PATH.exists(), (
        "Dataset not found. Run "
        "`python -m src.prepare_data` first."
    )

    return pd.read_csv(DATA_PATH)


def test_expected_columns_are_present():
    df = load_data()

    expected_columns = {
        "Age",
        "Attrition",
        "BusinessTravel",
        "Department",
        "DistanceFromHome",
        "Education",
        "Gender",
        "JobRole",
        "MonthlyIncome",
        "OverTime",
        "YearsAtCompany",
    }

    assert expected_columns.issubset(
        set(df.columns)
    )


def test_target_contains_expected_values():
    df = load_data()

    assert set(
        df[TARGET].dropna().unique()
    ).issubset({"Yes", "No"})


def test_numeric_ranges_are_reasonable():
    df = load_data()

    assert df["Age"].dropna().between(
        18, 80
    ).all()

    assert (
        df["MonthlyIncome"]
        .dropna()
        .gt(0)
        .all()
    )

    assert (
        df["DistanceFromHome"]
        .dropna()
        .ge(0)
        .all()
    )


def test_dataset_has_required_size():
    df = load_data()

    assert len(df) >= 1000
    assert len(df.columns) >= 8


def test_dataset_has_numeric_and_categorical_columns():
    df = load_data()

    numeric = df.select_dtypes(
        include="number"
    )

    categorical = df.select_dtypes(
        include=["object", "category"]
    )

    assert len(numeric.columns) >= 1
    assert len(categorical.columns) >= 1


def test_dataset_contains_missing_values():
    df = load_data()

    assert df.isna().sum().sum() > 0
