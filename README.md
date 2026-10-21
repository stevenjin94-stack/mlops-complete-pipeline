# Complete MLOps Pipeline

An end-to-end MLOps project for predicting employee attrition using the IBM HR Analytics Employee Attrition dataset.

The project demonstrates:

- Git and GitHub version control
- DVC dataset versioning
- YAML-based configuration
- scikit-learn preprocessing and modeling
- MLflow experiment tracking
- pytest automated testing
- GitHub Actions CI/CD
- Evidently data drift monitoring

---

## Project Overview

The prediction task is binary classification.

The model predicts whether an employee is likely to leave the organization.

Target:

`Attrition`

Possible values:

- `Yes`
- `No`

The IBM HR Analytics dataset contains approximately 1,470 employee records and 35 columns.

The dataset includes both numeric and categorical variables.

The original dataset does not contain missing values, so this project reproducibly injects a small percentage of missing values to demonstrate missing-data handling.

---

## Repository Structure

```text
mlops-complete-pipeline/
├── .github/workflows/
│   └── ci.yml
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
├── src/
│   ├── data_preprocessing.py
│   ├── prepare_data.py
│   ├── evaluate.py
│   ├── train.py
│   ├── run_experiments.py
│   ├── compare_experiments.py
│   └── monitor_drift.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_data_validation.py
│   └── test_model.py
├── configs/config.yaml
├── requirements.txt
├── README.md
└── MONITORING.md

---

## Environment Setup

Use Python 3.11 for the most consistent results with the project's GitHub Actions workflow.

### 1. Clone the repository

```bash
git clone https://github.com/stevenjin94-stack/mlops-complete-pipeline.git
cd mlops-complete-pipeline
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, consult the Python documentation for your system's execution-policy settings.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Retrieve the dataset using DVC

The raw employee attrition dataset is managed by DVC rather than stored directly in Git.

The project's default DVC remote is DagsHub (`origin`). If access credentials are required, configure them locally according to the DagsHub instructions. Do not commit credentials or `.dvc/config.local` to Git.

Retrieve the tracked dataset:

```powershell
dvc pull
```

Verify that the local cache and default remote are synchronized:

```powershell
dvc status -c
```

If `dvc pull` fails, verify remote access and credentials before proceeding.

### 5. Prepare the data

```powershell
python -m src.prepare_data
```

This prepares the dataset for model training using the project's configured preprocessing workflow.

### 6. Train the model

```powershell
python -m src.train --config configs/config.yaml
```

Training uses a validation set to select the classification threshold and reserves the test set for final evaluation. MLflow records experiment parameters and metrics.

### 7. Run configured experiments

```powershell
python -m src.run_experiments
```

Compare experiment results:

```powershell
python -m src.compare_experiments
```

Experiment comparison uses validation F1 to select the best run and reports test metrics separately.

### 8. Run data drift monitoring

```powershell
python -m src.monitor_drift --config configs/config.yaml
```

The monitoring script compares reference data with simulated production data, reports the overall drift share and individual drifted features, and generates an HTML report under `reports/`.

### 9. Run the automated tests

```powershell
python -m pytest tests/ -v
```

The test suite covers data validation, preprocessing, and model behavior.

### 10. Run the full local workflow

From the activated virtual environment, run:

```powershell
dvc pull
python -m src.prepare_data
python -m pytest tests/ -v
python -m src.train --config configs/config.yaml
python -m src.monitor_drift --config configs/config.yaml
```

### Experiment tracking

MLflow is used to record training runs, parameters, and evaluation metrics. The local tracking configuration is determined by the project's training code and MLflow setup.

### Reproducibility notes

- Use the same Python and dependency versions documented in `requirements.txt`.
- Retrieve the DVC-managed dataset before running data preparation or training.
- Keep credentials in local configuration or environment variables, never in committed files.
- The CI workflow runs automated tests and model training on pushes and pull requests targeting `main`.
