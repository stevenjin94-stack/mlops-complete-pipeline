from pathlib import Path

import numpy as np
import pandas as pd


DATA_URL = (
    "https://raw.githubusercontent.com/"
    "nelson-wu/employee-attrition-ml/master/"
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

OUTPUT_PATH = Path("data/raw/employee_attrition.csv")


def download_dataset(output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    """Download the IBM HR attrition dataset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_URL)

    if df.shape[0] < 1000:
        raise ValueError("Dataset must contain at least 1,000 rows.")

    if df.shape[1] < 8:
        raise ValueError("Dataset must contain at least 8 columns.")

    df.to_csv(output_path, index=False)

    return df


def inject_missing_values(
    df: pd.DataFrame,
    fraction: float = 0.02,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Add reproducible missing values to selected feature columns.

    The original dataframe is never modified.
    """
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1.")

    result = df.copy(deep=True)

    feature_columns = [
        "MonthlyIncome",
        "JobSatisfaction",
        "DistanceFromHome",
        "JobRole",
        "OverTime",
    ]

    rng = np.random.default_rng(random_state)

    for column in feature_columns:
        if column not in result.columns:
            raise ValueError(f"Required column '{column}' is missing.")

        n_missing = max(1, int(len(result) * fraction))
        indices = rng.choice(
            result.index,
            size=n_missing,
            replace=False,
        )

        result.loc[indices, column] = np.nan

    return result


def main() -> None:
    df = download_dataset()

    df_with_missing = inject_missing_values(
        df,
        fraction=0.02,
        random_state=42,
    )

    df_with_missing.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved dataset to {OUTPUT_PATH}")
    print(f"Shape: {df_with_missing.shape}")
    print(f"Missing values: {df_with_missing.isna().sum().sum()}")


if __name__ == "__main__":
    main()
