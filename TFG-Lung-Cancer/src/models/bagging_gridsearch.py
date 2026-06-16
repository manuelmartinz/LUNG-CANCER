"""
bagging_gridsearch.py — Grid search secuencial para Bagging.
"""

import sys
import os
import argparse
import wandb
import numpy as np

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.datareader import DataReader
from src.grid_search import SequentialGridSearch


def get_param_grid() -> dict:
    """Rejilla de hiperparámetros para BaggingClassifier."""
    return {
        "n_estimators": [10, 25, 50, 100, 200],
        "max_samples": [0.5, 0.7, 0.9, 1.0],
        "max_features": [0.5, 0.7, 0.9, 1.0],
        "bootstrap": [True, False],
        "bootstrap_features": [False, True],
        
    }


def main():
    parser = argparse.ArgumentParser(description="Bagging — Grid Search Secuencial")
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
            name="Bagging-sequential-gs",
            config={
                "model": "Bagging",
                "param_grid": {k: [str(v) for v in vals] for k, vals in get_param_grid().items()},
                **reader.get_config(),
            },
        )

    base_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    class_weight=None,
    random_state=args.seed
)
    estimator = BaggingClassifier(
        estimator=base_tree,
        random_state=args.seed,
        n_jobs=-1,
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

    if hasattr(gs.best_estimator_, "predict_proba"):
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

    gs.save_results("results/bagging")


if __name__ == "__main__":
    main()
