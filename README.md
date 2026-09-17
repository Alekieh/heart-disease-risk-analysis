# Heart Disease Risk Analysis

A complete end-to-end data science project exploring and modelling heart disease risk from patient health data.

## Central question

> Do lifestyle features (sleep, stress, daily steps, diet, exercise) add genuine predictive power on top of clinical labs when predicting heart disease?

## Dataset

`heart_disease_risk_2026.csv` — 9,001 patient records, 26 features, binary target (`has_heart_disease`).

> **Note:** This is a synthetic dataset. All findings are for learning/portfolio purposes and cannot be used for diagnosis or clinical decision-making.

## Project structure

`
.
+-- heart_disease_risk_2026.csv          # Raw data
+-- heart_disease_risk_analysis.ipynb   # Main notebook (EDA + modelling + SHAP)
+-- requirements.txt                    # Python dependencies
+-- README.md
`

## Notebook contents

| Section | Description |
|---|---|
| 0 · Setup | Imports, data load, sanity checks |
| 1 · EDA | Class balance, correlation analysis, 3 targeted research questions |
| 2 · Preprocessing | Encoding, scaling, 60/20/20 stratified split |
| 3 · Modelling | Logistic Regression, Random Forest, XGBoost — ROC-AUC, precision, recall, F1 |
| 4 · SHAP & Ablation | Feature importance, ablation experiment (lifestyle vs clinical), dependence plots |
| 5 · Write-up | Findings, limitations, disclaimer |

## Key findings

- **Clinical features drive most of the signal** — ST depression, HbA1c, and age are the top predictors.
- **Lifestyle features add independent value** — the full model outperforms clinical-only on ROC-AUC, and partial correlations confirm stress score and sleep hours carry signal beyond BP/cholesterol.
- **Wearable ownership alone is not a reliable proxy** for healthier behaviour in this dataset.
- **Risk scales monotonically with age** and is compounded by smoking and family history.

## Setup

`ash
# Create environment
uv venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Install dependencies
uv pip install -r requirements.txt

# Launch notebook
jupyter lab
`

## Tech stack

Python · pandas · scikit-learn · XGBoost · SHAP · seaborn · matplotlib · scipy
