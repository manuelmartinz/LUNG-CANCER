"""
logistic_regression_gridsearch.py — Grid search secuencial para Regresión Logística.
"""

import sys
import os
import argparse
import wandb
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix , roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.datareader import DataReader
from src.grid_search import SequentialGridSearch


def get_param_grid() -> dict:
    """Rejilla de hiperparámetros para Regresión Logística."""
    return {
        "C": [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
        "solver": ["lbfgs", "liblinear", "saga"],
        "class_weight": [None, "balanced"],
        "max_iter": [200, 500, 1000, 2000],
    }


def main():
    parser = argparse.ArgumentParser(description="Logistic Regression — Grid Search Secuencial")
    parser.add_argument("--data", type=str, default="data/lung_cancer_dataset.csv")
    parser.add_argument("--encoding", type=str, default="onehot")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    reader = DataReader(
        filepath=args.data,
        target_col="lung_cancer",
        drop_cols=["patient_id"],
        encoding=args.encoding,
        random_state=args.seed,
    )
    X_train, X_test, y_train, y_test = reader.load_split()


    

    run = None
    if not args.no_wandb:
        run = wandb.init(
            project="TFG-clasificacion",
            name="LogisticRegression-sequential-gs",
            config={
                "model": "LogisticRegression",
                "param_grid": {k: [str(v) for v in vals] for k, vals in get_param_grid().items()},
                **reader.get_config(),
            },
        )

    estimator = LogisticRegression(random_state=args.seed)
    gs = SequentialGridSearch(
        estimator=estimator,
        param_grid=get_param_grid(),
        scoring="f1_weighted",
        cv_splits=5,
        random_state=args.seed,
        wandb_run=run,
    )
    gs.fit(X_train, y_train)

    y_pred = gs.best_estimator_.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Evaluación en Test ===")
    print(classification_report(y_test, y_pred))
    print("Matriz de confusión:")
    print(cm)
    print(f"Mejores hiperparámetros: {gs.best_params_}")
    print(f"Accuracy en test: {report['accuracy']:.4f}")
    print(f"ROC AUC en test: {roc_auc_score(y_test, gs.best_estimator_.predict_proba(X_test)[:, 1]):.4f}")

    gs.set_test_metrics({
        "test_accuracy": report["accuracy"],
        "test_f1_weighted": report["weighted avg"]["f1-score"],
        "test_precision_weighted": report["weighted avg"]["precision"],
        "test_recall_weighted": report["weighted avg"]["recall"],
    })
    

    if run is not None:
        run.log({
            "test/accuracy": report["accuracy"],
            "test/f1_weighted": report["weighted avg"]["f1-score"],
            "test/precision_weighted": report["weighted avg"]["precision"],
            "test/recall_weighted": report["weighted avg"]["recall"],
        })
        run.summary["best_params"] = gs.best_params_
        run.summary["best_val_score"] = gs.best_score_
        run.finish()

    gs.save_results("results/logistic_regression")


if __name__ == "__main__":
    main()
