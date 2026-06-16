"""
voting_gridsearch.py — Grid search secuencial para Voting Classifier.

El Voting Classifier combina varios estimadores base. Aquí se optimizan
los hiperparámetros de cada estimador base y el tipo de votación.
"""

import sys
import os
import argparse
import wandb
import numpy as np

from sklearn.ensemble import (
    VotingClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
)

from catboost import CatBoostClassifier

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.datareader import DataReader
from src.grid_search import SequentialGridSearch


def get_param_grid() -> dict:
    """Rejilla de hiperparámetros para Voting Classifier."""
    return {
        "voting": ["hard", "soft"],
        
    }


def main():
    parser = argparse.ArgumentParser(description="Voting Classifier — Grid Search Secuencial")
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
            name="VotingClassifier-sequential-gs",
            config={
                "model": "VotingClassifier",
                "param_grid": {k: [str(v) for v in vals] for k, vals in get_param_grid().items()},
                **reader.get_config(),
            },
        )

    estimator = VotingClassifier(
        estimators=[
            ("ada", AdaBoostClassifier(random_state=args.seed, n_estimators=500, learning_rate=0.05)),
            ("histgb", HistGradientBoostingClassifier(random_state=args.seed, max_iter=100, max_depth=3, 
                                                  learning_rate=0.2, min_samples_leaf=1, max_leaf_nodes=15, 
                                                  l2_regularization=1.0, max_bins=128)),


            ("xgb", XGBClassifier(random_state=args.seed, n_estimators=25, max_depth=3, learning_rate=0.3, 
                                  subsample=0.5, colsample_bytree=1.0)),

            ("cat", CatBoostClassifier(random_state=args.seed, iterations=100, depth=4, learning_rate=0.2, l2_leaf_reg=1, border_count=64,verbose=0, allow_writing_files=False)),
        ],
        voting="soft",
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
    if(gs.best_params_["voting"] == "soft"):
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

    gs.save_results("results/voting")


if __name__ == "__main__":
    main()
