# Week 5 Interim Submission

## 1. Introduction
This project builds a proxy-based credit scoring model for Bati Bank’s BNPL service.

## 2. Project Structure
- `data/raw/`: raw CSV data  
- `data/processed/`: processed data outputs (`with_labels.csv`, `features.csv`, `labels.csv`)  
- `notebooks/1.0-eda.ipynb`: exploratory data analysis  
- `src/data_processing.py`: RFM label creation & feature pipeline  
- `src/train.py`: (to be implemented) model training  
- `src/api/`: (to be implemented) FastAPI service  
- `tests/`: (to be implemented) unit tests  
- `.github/workflows/ci.yml`: CI/CD configuration  
- `Dockerfile` & `docker-compose.yml`: containerization setup  

## 3. Credit Scoring Business Understanding
### 1. Basel II and Interpretability
Basel II mandates transparent, auditable risk models, requiring explainable methodologies and detailed documentation to satisfy regulatory scrutiny.

### 2. Proxy Variable Necessity & Risks
In absence of real default labels, we create a proxy by labeling disengaged customers (via RFM clustering) as high-risk. This may misclassify customers, leading to revenue loss or unexpected defaults.

### 3. Simple vs. Complex Models
- **Simple (Logistic Regression + WoE):** Easy to explain, fast, but may underfit complex patterns.  
- **Complex (Gradient Boosting):** High accuracy, captures nonlinearities, but “black-box” nature complicates compliance.

## 4. Exploratory Data Analysis (EDA) Insights
1. No missing values detected across all columns.  
2. Transaction amounts are highly right-skewed (median = 1 000, max = 9 880 000).  
3. Amount and Value are near perfectly correlated—one is redundant.  
4. ChannelId is imbalanced (two major channels vs. very rare others).  
5. FraudResult is extremely imbalanced (≈ 99.8% non-fraud).

## 5. Proxy Target Engineering
- Computed RFM metrics (Recency, Frequency, Monetary) per customer.  
- Scaled features and ran KMeans (3 clusters).  
- Labeled the cluster with highest Recency & lowest engagement as `is_high_risk`.  
- Implemented both in the notebook and automated in `src/data_processing.py`.

## 6. Feature Engineering Pipeline
- Extracted datetime features (hour, day, month, year).  
- Ordinal-encoded all categorical features.  
- Median-imputed & StandardScaled numerical features (`Amount`, `Value`).  
- Outputs saved to `data/processed/features.csv` and `labels.csv`.

## 7. Next Steps
- Train and evaluate models (Logistic Regression, Random Forest, etc.).  
- Track experiments with MLflow and implement unit tests with pytest.  
- Build and containerize FastAPI service; configure GitHub Actions for CI/CD.  
