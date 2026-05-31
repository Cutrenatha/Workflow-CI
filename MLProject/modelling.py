import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import os
import logging
import json
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, log_loss
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── DagsHub / MLflow ─────────────────────────────────────────────────────────
DAGSHUB_USER = os.getenv("DAGSHUB_USER", "nama_owner")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO", "nama_repo")
dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)
mlflow.set_experiment("Credit Scoring - CI")


# ─── Plot Helpers ─────────────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    path = "training_confusion_matrix.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


def save_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(indices)), importances[indices], color="#4A90D9")
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    ax.set_title("Feature Importances")
    ax.set_ylabel("Gini Importance")
    path = "feature_importance.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    logger.info("Loading data...")
    train_df = pd.read_csv(args.train_path)
    test_df  = pd.read_csv(args.test_path)

    X_train = train_df.drop(columns=[args.target_col])
    y_train = train_df[args.target_col]
    X_test  = test_df.drop(columns=[args.target_col])
    y_test  = test_df[args.target_col]

    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    with mlflow.start_run(run_name="RF_CI_Run") as run:
        # ── Params ────────────────────────────────────────────────────────────
        mlflow.log_params({
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "random_state": args.random_state,
        })

        # ── Train ─────────────────────────────────────────────────────────────
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            n_jobs=-1,
        )
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        # ── Predict ───────────────────────────────────────────────────────────
        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # ── Metrics ───────────────────────────────────────────────────────────
        acc     = accuracy_score(y_test, y_pred)
        prec    = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec     = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1      = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        auc     = roc_auc_score(y_test, y_pred_prob)
        logloss = log_loss(y_test, model.predict_proba(X_test))

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted")

        mlflow.log_metrics({
            "accuracy": acc,
            "precision_weighted": prec,
            "recall_weighted": rec,
            "f1_score_weighted": f1,
            "roc_auc": auc,
            "log_loss": logloss,
            "training_time_sec": round(elapsed, 2),
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
        })
        logger.info(f"Metrics: Acc={acc:.4f} F1={f1:.4f} AUC={auc:.4f}")

        # ── Model ─────────────────────────────────────────────────────────────
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="CreditScoringRF_CI",
        )

        # ── Artifacts ─────────────────────────────────────────────────────────
        mlflow.log_artifact(save_confusion_matrix(y_test, y_pred))
        mlflow.log_artifact(save_feature_importance(model, list(X_train.columns)))

        report = classification_report(y_test, y_pred, output_dict=True)
        with open("classification_report.json", "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact("classification_report.json")

        # Save model locally for Docker
        os.makedirs("model", exist_ok=True)
        joblib.dump(model, "model/model.pkl")
        mlflow.log_artifact("model/model.pkl", artifact_path="model")

        # Write run_id for downstream steps
        with open("latest_run_id.txt", "w") as f:
            f.write(run.info.run_id)

        logger.info(f"Run ID: {run.info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path",   default="./credit_scoring_preprocessing/train.csv")
    parser.add_argument("--test_path",    default="./credit_scoring_preprocessing/test.csv")
    parser.add_argument("--target_col",   default="Risk")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth",    type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    main(args)
