from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from evidently import Report
from evidently.presets import DataDriftPreset


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_production_data(
    reference: pd.DataFrame,
    fraction: float,
    random_state: int,
) -> pd.DataFrame:
    """
    Create simulated production data with intentional drift.
    """
    production = reference.sample(
        frac=fraction,
        random_state=random_state,
    ).copy()

    # Introduce realistic numerical drift.
    production["Age"] = (
        production["Age"] + 8
    ).clip(lower=18)

    production["MonthlyIncome"] = (
        production["MonthlyIncome"] * 1.20
    )

    # Increase overtime frequency.
    production["OverTime"] = "Yes"

    return production


def calculate_drift_share(report_dict: dict) -> float:
    """Extract overall drift share from Evidently output."""

    metrics = report_dict.get("metrics", [])

    for metric in metrics:
        metric_id = metric.get("metric_id", "")

        if "DriftedColumnsCount" in metric_id:
            value = metric.get("value", {})

            if isinstance(value, dict):
                drifted = value.get("count", 0)
                total = value.get(
                    "total",
                    value.get("share", 0),
                )

                if isinstance(total, (int, float)) and total:
                    return drifted / total

    # Fallback for schema variations.
    for metric in metrics:
        value = metric.get("value")

        if isinstance(value, dict):
            if "share" in value:
                return float(value["share"])

    return 0.0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    data_path = config["data"]["raw_path"]

    reference = pd.read_csv(data_path)

    production = create_production_data(
        reference,
        fraction=config["monitoring"][
            "production_sample_fraction"
        ],
        random_state=config["monitoring"][
            "random_state"
        ],
    )

    # Target is excluded from feature drift monitoring.
    target = config["data"]["target_column"]

    reference_features = reference.drop(
        columns=[target]
    )

    production_features = production.drop(
        columns=[target]
    )


    report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )

    snapshot = report.run(
        reference_data=reference_features,
        current_data=production_features,
    )

    output_dir = Path("reports")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / "drift_report.html"
    snapshot.save_html(str(output_path))

    result = snapshot.dict()
    drift_share = calculate_drift_share(result)

    print("\nDrift Monitoring Summary")
    print("=" * 60)
    print(f"Overall drift share: {drift_share:.2%}")
    print(f"HTML report: {output_path}")

    threshold = config["monitoring"]["drift_threshold"]

    if drift_share > threshold:
        print(
            f"DRIFT THRESHOLD EXCEEDED: "
            f"{drift_share:.2%} > {threshold:.2%}"
        )
        raise SystemExit(1)

    print(f"Drift is below threshold ({threshold:.2%}).")


if __name__ == "__main__":
    main()
