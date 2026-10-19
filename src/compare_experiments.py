
import mlflow
import pandas as pd
import yaml


def main():
    with open(
        "configs/config.yaml",
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    mlflow.set_tracking_uri(
        config["training"]["mlflow_tracking_uri"]
    )

    experiment = mlflow.get_experiment_by_name(
        config["training"]["experiment_name"]
    )

    if experiment is None:
        raise RuntimeError(
            "MLflow experiment does not exist. "
            "Run the experiments first."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1 DESC"],
    )

    if runs.empty:
        raise RuntimeError("No MLflow runs were found.")

    # Exclude runs without valid F1 scores.
    runs["metrics.f1"] = pd.to_numeric(
        runs["metrics.f1"], errors="coerce"
    )
    runs = runs.dropna(subset=["metrics.f1"])

    if "status" in runs.columns:
        runs = runs[runs["status"] == "FINISHED"]

    # Match each configured experiment to its parameters.
    selected = []

    for item in config["experiments"]:
        candidates = runs[
            (runs["params.n_estimators"].astype(str)
             == str(item["n_estimators"]))
            & (
                runs["params.max_depth"].fillna("None").astype(str)
                == str(item["max_depth"])
            )
        ].copy()

        if candidates.empty:
            print(
                f"WARNING: No completed run found for "
                f"{item['name']} "
                f"(n_estimators={item['n_estimators']}, "
                f"max_depth={item['max_depth']})"
            )
            continue

        # If a configuration has been run more than once,
        # retain its highest-F1 completed run.
        best_candidate = candidates.loc[
            candidates["metrics.f1"].idxmax()
        ].copy()

        best_candidate["configured_name"] = item["name"]
        selected.append(best_candidate)

    if not selected:
        raise RuntimeError(
            "No configured experiments could be matched "
            "to valid MLflow runs."
        )

    results = pd.DataFrame(selected).sort_values(
        "metrics.f1", ascending=False
    )

    columns = [
        "configured_name",
        "run_id",
        "metrics.f1",
        "metrics.accuracy",
        "metrics.precision",
        "metrics.recall",
        "params.n_estimators",
        "params.max_depth",
    ]

    print("\nConfigured Experiment Results")
    print("=" * 90)
    print(results[columns].to_string(index=False))

    best = results.iloc[0]

    print("\nBest Run")
    print("=" * 90)
    print(f"Experiment: {best['configured_name']}")
    print(f"Run ID: {best['run_id']}")
    print(f"F1: {best['metrics.f1']:.4f}")

    if pd.notna(best.get("metrics.accuracy")):
        print(f"Accuracy: {best['metrics.accuracy']:.4f}")


if __name__ == "__main__":
    main()
