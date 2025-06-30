# kaim-5-week-5

[![CI](https://github.com/HTGit63/kaim-5-week-5/actions/workflows/ci.yml/badge.svg)](https://github.com/HTGit63/kaim-5-week-5/actions)

**Credit Risk Probability Model**  
_End-to-end implementation of a credit scoring system for Bati Bank’s buy-now-pay-later service._

---

## Table of Contents

1. [Overview](#overview)  
2. [Credit Scoring Business Understanding](#credit-scoring-business-understanding)  
3. [Setup](#setup)  
4. [Usage](#usage)  
   - [Data Processing](#data-processing)  
   - [Model Training](#model-training)  
   - [Model Evaluation](#model-evaluation)  
   - [API Service](#api-service)  
5. [Project Structure](#project-structure)  
6. [CI/CD Pipeline](#cicd-pipeline)  
7. [References](#references)  

---

## Overview

This repository implements a proxy-based credit scoring model using eCommerce transaction data from the Xente platform. We engineer RFM-based features, define a binary "high-risk" proxy label via clustering, train and tune two ML models, select the best, and deploy it as a FastAPI service with CI/CD.

## Credit Scoring Business Understanding

- **Basel II regulatory context:**  
  The Basel II Accord requires banks to rigorously measure and document credit risk. Our model emphasizes interpretability (via RFM proxy construction) and traceability to satisfy regulatory auditability.

- **Proxy target necessity:**  
  Without a ground-truth default label, we use RFM clustering to identify disengaged (high Recency, low Frequency, low Monetary) customers as a high-risk proxy.  **Business risk:** proxy may misclassify genuine low-engagement customers as high-risk, affecting customer experience.

- **Model trade-offs in finance:**  
  - **Logistic Regression + WoE:** interpretable, easier regulatory approval, but lower performance.  
  - **Gradient-boosted trees / Random Forest:** higher ROC‑AUC and recall, but more complex and harder to fully explain.  
  We selected Random Forest (ROC‑AUC = 0.926) to balance performance and acceptable interpretability via feature importance.

## Setup

```bash
git clone https://github.com/HTGit63/kaim-5-week-5.git
cd kaim-5-week-5
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Data Processing

```bash
python src/data_processing.py
```
- Reads `data/raw/data.csv`, computes RFM features, clusters customers, labels high-risk, extracts datetime + categorical features, and writes:
  - `data/processed/features.csv`
  - `data/processed/labels.csv`

### Model Training

```bash
python src/train.py
```
- Splits data, runs GridSearchCV for LogisticRegression and RandomForest, logs experiments in MLflow, outputs:
  - `experiments/results.csv`
  - `models/best_model.pkl`

### Model Evaluation

See `notebooks/2.0-model-evaluation.ipynb` for:
- Metrics comparison table  
- ROC‑AUC bar chart  
- ROC curve for the final model  

### API Service

Run locally:

```bash
uvicorn src.api.main:app --reload
```

**Endpoint:** `POST /predict`  
**Request JSON:**  
```json
{
  "BatchId": 46980.0,
  "AccountId": 2490.0,
  "SubscriptionId": 3535.0,
  "CurrencyCode": 0.0,
  "ProviderId": 5.0,
  "ProductId": 1.0,
  "ProductCategory": 0.0,
  "ChannelId": 2.0,
  "PricingStrategy": 2.0,
  "Amount": -0.046,
  "Value": -0.072,
  "hour": 2,
  "day": 15,
  "month": 11,
  "year": 2018
}
```

**Response JSON:**
```json
{
  "risk_probability": 0.12,
  "is_high_risk": 0
}
```

## Project Structure

```
credit-risk-model/
├── data/
│   ├── raw/                 # Raw CSV files
│   └── processed/           # Features and labels output
├── notebooks/
│   ├── 1.0-eda.ipynb        # Exploratory Data Analysis
│   └── 2.0-model-evaluation.ipynb # Model comparison & selection
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI app
│   │   └── pydantic_models.py
│   ├── data_processing.py   # RFM + feature pipeline
│   ├── train.py             # Model training & MLflow logging
│   └── __init__.py
├── tests/
│   └── test_api.py          # API endpoint smoke test
├── .github/workflows/ci.yml # GitHub Actions CI
├── Dockerfile               # Containerization (API)
├── docker-compose.yml       # Docker Compose
├── requirements.txt         # Dev and ML dependencies
├── requirements-api.txt     # Lean API dependencies
└── README.md
```

## CI/CD Pipeline

- **Linting:** `flake8 src tests`  
- **Testing:** `pytest` (smoke test for `/predict`)  
- Runs on every push and pull request via GitHub Actions.
