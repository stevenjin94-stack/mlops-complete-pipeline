from src.train import load_config, train


def main():
    config_path = "configs/config.yaml"
    config = load_config(config_path)

    experiments = config["experiments"]

    print(
        f"Running {len(experiments)} MLflow experiments..."
    )

    for experiment in experiments:
        name = experiment["name"]

        print("\n" + "=" * 60)
        print(f"Experiment: {name}")
        print("=" * 60)

        overrides = {
            "n_estimators": experiment["n_estimators"],
            "max_depth": experiment["max_depth"],
        }

        result = train(
            config_path=config_path,
            overrides=overrides,
        )

        print(
            f"{name}: "
            f"F1={result['f1']:.4f}, "
            f"Accuracy={result['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
