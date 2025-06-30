import os
import pandas as pd
import mlflow
import mlflow.sklearn
from joblib import dump
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from mlflow.models.signature import infer_signature


def load_data(
    features_path="data/processed/features.csv",
    labels_path="data/processed/labels.csv",
):
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path).iloc[:, 0]
    return X, y


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def main():
    # 1. Load and clean data
    X, y = load_data()
    df = pd.concat([X, y.rename("is_high_risk")], axis=1).dropna().reset_index(drop=True)
    X = df.drop(columns=["is_high_risk"])
    y = df["is_high_risk"].astype(int)

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. MLflow setup
    mlflow.set_experiment("credit_risk_experiments")

    # 4. Models + hyperparameter grids
    models = {
        "LogisticRegression": {
            "estimator": LogisticRegression(solver="liblinear", max_iter=1000),
            "params": {"C": [0.01, 0.1, 1, 10], "penalty": ["l1", "l2"]},
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(random_state=42),
            "params": {"n_estimators": [50, 100], "max_depth": [5, 10, None]},
        },
    }

    best_overall_auc = -1.0
    best_overall_model = None
    results = []

    # 5. GridSearch + MLflow logging
    for name, spec in models.items():
        with mlflow.start_run(run_name=name):
            gs = GridSearchCV(
                spec["estimator"],
                spec["params"],
                cv=3,
                scoring="roc_auc",
                n_jobs=-1,
            )
            gs.fit(X_train, y_train)
            best = gs.best_estimator_
            perf = evaluate_model(best, X_test, y_test)

            # Log configuration & metrics
            mlflow.log_param("model_name", name)
            mlflow.log_params(gs.best_params_)

            # infer signature so downstream consumers know the input/output schema
            sig = infer_signature(X_train, best.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=best,
                name="model",
                signature=sig,
                input_example=X_train.iloc[:5, :],
            )

            mlflow.log_metrics(perf)

            # Track best
            if perf["roc_auc"] > best_overall_auc:
                best_overall_auc = perf["roc_auc"]
                best_overall_model = best

            # record for local CSV
            results.append({"model": name, **gs.best_params_, **perf})

    # 6. Save per-model results
    os.makedirs("experiments", exist_ok=True)
    pd.DataFrame(results).to_csv("experiments/results.csv", index=False)

    # 7. Retrain best model on full data & persist
    if best_overall_model is not None:
        best_overall_model.fit(X, y)
        os.makedirs("models", exist_ok=True)
        dump(best_overall_model, "models/best_model.pkl")
        print("✅ Training complete.")
        print("Results table → experiments/results.csv")
        print("Best model saved → models/best_model.pkl")
    else:
        print("⚠️  No model was trained. Check your data and hyperparameters.")


if __name__ == "__main__":
    main()
