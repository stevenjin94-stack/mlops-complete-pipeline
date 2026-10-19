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
