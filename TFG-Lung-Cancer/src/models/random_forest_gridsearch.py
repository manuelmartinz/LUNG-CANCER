"""
random_forest_gridsearch.py — Grid search secuencial para Random Forest.
"""

import sys
import os
import argparse
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix , roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.datareader import DataReader
from src.grid_search import SequentialGridSearch


def get_param_grid() -> dict:
    """Rejilla de hiperparámetros para Random Forest (orden secuencial)."""
    return {
        "n_estimators": [50, 100, 200, 300, 500],
        "max_depth": [5, 10, 15, 20, 30, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
        "criterion": ["gini", "entropy"],
        "class_weight": [None, "balanced", "balanced_subsample"],
    }


def main():
    parser = argparse.ArgumentParser(description="Random Forest — Grid Search Secuencial")
    parser.add_argument("--data", type=str, default="data/lung_cancer_dataset.csv")
    parser.add_argument("--encoding", type=str, default="onehot")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Cargar datos
    reader = DataReader(
        filepath=args.data,
        target_col="lung_cancer",
        drop_cols=["patient_id"],
        encoding=args.encoding,
        random_state=args.seed,
        scale=False,
    )
    X_train, X_test, y_train, y_test = reader.load_split()

    # wandb
    run = None
    if not args.no_wandb:
        run = wandb.init(
            project="TFG-clasificacion",
            name="RandomForest-sequential-gs",
            config={
                "model": "RandomForest",
                "param_grid": {k: [str(v) for v in vals] for k, vals in get_param_grid().items()},
                **reader.get_config(),
            },
        )

    # Grid search secuencial
    estimator = RandomForestClassifier(random_state=args.seed, n_jobs=-1)
    gs = SequentialGridSearch(
        estimator=estimator,
        param_grid=get_param_grid(),
        scoring="accuracy",
        cv_splits=5,
        random_state=args.seed,
        wandb_run=run,
    )
    gs.fit(X_train, y_train)

    # Evaluación final en test
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

    bmodel = gs.best_estimator_

    importances = bmodel.feature_importances_

    df_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": importances
    }).sort_values(by="importance", ascending=False)


    top_n = 15
    df_top = df_importance.head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(df_top["feature"], df_top["importance"])
    plt.gca().invert_yaxis()
    plt.title("Top 15 Feature Importance - Random Forest")
    plt.xlabel("Importance")

    plt.savefig("feature_importance_randomforest.png", dpi=300, bbox_inches="tight")
    plt.close()


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

    # Guardar CSVs
    gs.save_results("results/random_forest")


if __name__ == "__main__":
    main()
