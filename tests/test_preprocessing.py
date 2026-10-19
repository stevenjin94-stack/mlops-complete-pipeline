import pandas as pd
import pytest

from src.data_preprocessing import (
    build_preprocessor,
    split_features_target,
    validate_dataframe,
)


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "Age": [25, 30, None, 45],
            "Income": [3000, None, 5000, 7000],
            "Department": [
                "Sales",
                "HR",
                "Sales",
                None,
            ],
            "Attrition": [
                "No",
                "Yes",
                "No",
                "Yes",
            ],
        }
    )


def test_missing_values_are_handled(sample_dataframe):
    X, _ = split_features_target(
        sample_dataframe,
        "Attrition",
    )

    preprocessor = build_preprocessor(X)

    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == len(
        sample_dataframe
    )


def test_categorical_variables_are_encoded(
    sample_dataframe,
):
    X, _ = split_features_target(
        sample_dataframe,
        "Attrition",
    )

    preprocessor = build_preprocessor(X)

    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[1] > X.shape[1]


def test_original_dataframe_is_not_modified(
    sample_dataframe,
):
    original = sample_dataframe.copy(
        deep=True
    )

    split_features_target(
        sample_dataframe,
        "Attrition",
    )

    pd.testing.assert_frame_equal(
        sample_dataframe,
        original,
    )


def test_missing_target_raises_error(
    sample_dataframe,
):
    with pytest.raises(ValueError):
        split_features_target(
            sample_dataframe,
            "NotAColumn",
        )


def test_non_dataframe_raises_error():
    with pytest.raises(TypeError):
        validate_dataframe(
            ["not", "a", "dataframe"],
            "Attrition",
        )


def test_empty_dataframe_raises_error():
    empty = pd.DataFrame()

    with pytest.raises(ValueError):
        validate_dataframe(
            empty,
            "Attrition",
        )


def test_missing_feature_columns_raise_error():
    df = pd.DataFrame(
        {
            "Age": [20, 30],
            "Attrition": ["No", "Yes"],
        }
    )

    X, _ = split_features_target(
        df,
        "Attrition",
    )

    assert "Age" in X.columns
