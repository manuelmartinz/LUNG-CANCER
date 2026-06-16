"""
catboost_gridsearch.py — Grid search secuencial para CatBoost.
"""

import sys
import os
import argparse
import wandb
import numpy as np

from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.datareader import DataReader
from src.grid_search import SequentialGridSearch


def get_param_grid() -> dict:
    """Rejilla de hiperparámetros para CatBoost."""
    return {
        "iterations": [100, 200, 500, 800, 1000],
        "depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "l2_leaf_reg": [1, 3, 5, 7, 9],
        "border_count": [32, 64, 128, 254],
        "bagging_temperature": [0, 0.5, 1, 2],
    }


class AutoCatBoostClassifier(CatBoostClassifier):
    """Detecta columnas categóricas en pandas y las pasa a fit."""

    def fit(self, X, y=None, cat_features=None, **kwargs):
        if cat_features is None and hasattr(X, "select_dtypes"):
            cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
            if cat_cols:
                cat_features = [X.columns.get_loc(col) for col in cat_cols]
        return super().fit(X, y, cat_features=cat_features, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="CatBoost — Grid Search Secuencial")
    parser.add_argument("--data", type=str, default="data/lung_cancer_dataset.csv")
    parser.add_argument("--encoding", type=str, default="catboost")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    reader = DataReader(
        filepath=args.data,
        target_col="lung_cancer",
        drop_cols=["patient_id"],
        encoding=args.encoding,
        random_state=args.seed,
        scale=False,
    )
    X_train, X_test, y_train, y_test = reader.load_split()

    run = None
    if not args.no_wandb:
        run = wandb.init(
            project="TFG-clasificacion",
            name="CatBoost-sequential-gs",
            config={
                "model": "CatBoost",
                "param_grid": {k: [str(v) for v in vals] for k, vals in get_param_grid().items()},
                **reader.get_config(),
            },
        )

    estimator = AutoCatBoostClassifier(
        random_seed=args.seed,
        verbose=0,
        allow_writing_files=False,
    )
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

    gs.save_results("results/catboost")


if __name__ == "__main__":
    main()