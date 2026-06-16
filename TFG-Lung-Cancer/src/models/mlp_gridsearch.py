"""
mlp_gridsearch.py — Grid search secuencial para red neuronal con Keras/TensorFlow.
"""

import os

# Debe declararse antes de importar TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import wandb
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, recall_score

from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.datareader import DataReader
from src.grid_search import SequentialGridSearch


class KerasBinaryClassifier(BaseEstimator, ClassifierMixin):
    """Clasificador binario Keras compatible con la API de sklearn."""

    def __init__(
        self,
        units_1: int = 128,
        units_2: int = 64,
        units_3: int = 32,
        dropout_1: float = 0.3,
        dropout_2: float = 0.3,
        l2_reg: float = 1e-4,
        learning_rate: float = 5e-4,
        batch_size: int = 64,
        epochs: int = 200,
        validation_split: float = 0.2,
        patience_early: int = 15,
        patience_reduce_lr: int = 5,
        threshold: float = 0.5,
        random_state: int = 42,
        verbose: int = 0,
    ):
        self.units_1 = units_1
        self.units_2 = units_2
        self.units_3 = units_3
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.patience_early = patience_early
        self.patience_reduce_lr = patience_reduce_lr
        self.threshold = threshold
        self.random_state = random_state
        self.verbose = verbose

        self.model_: Optional[keras.Model] = None
        self.history_ = None
        self.classes_ = np.array([0, 1])

    def _build_model(self, n_features: int) -> keras.Model:
        model = keras.Sequential(
            [
                layers.Input(shape=(n_features,)),
                layers.Dense(
                    self.units_1,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                ),
                layers.BatchNormalization(),
                layers.Dropout(self.dropout_1),
                layers.Dense(
                    self.units_2,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                ),
                layers.BatchNormalization(),
                layers.Dropout(self.dropout_2),
                layers.Dense(
                    self.units_3,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                ),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X, y):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(self.random_state)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)

        self.model_ = self._build_model(X_arr.shape[1])
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.patience_early,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=self.patience_reduce_lr,
                min_lr=1e-6,
            ),
        ]

        self.history_ = self.model_.fit(
            X_arr,
            y_arr,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=self.verbose,
        )
        return self

    def predict_proba(self, X):
        if self.model_ is None:
            raise RuntimeError("El modelo aún no ha sido entrenado.")
        X_arr = np.asarray(X, dtype=np.float32)
        proba_pos = self.model_.predict(X_arr, verbose=0).reshape(-1)
        proba_neg = 1.0 - proba_pos
        return np.column_stack([proba_neg, proba_pos])

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)


def plot_accuracy(history_obj, output_path: str) -> None:
    """Guarda la curva de accuracy de entrenamiento y validación."""
    if history_obj is None:
        return

    acc = history_obj.history.get("accuracy", [])
    val_acc = history_obj.history.get("val_accuracy", [])
    if not acc or not val_acc:
        return

    epochs = range(1, len(acc) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, acc, label="Train Accuracy", marker="o")
    plt.plot(epochs, val_acc, label="Val Accuracy", marker="o")
    plt.title("Evolucion Accuracy")
    plt.xlabel("Epoca")
    plt.ylabel("Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def get_param_grid() -> dict:
    """Rejilla de hiperparámetros para red neuronal Keras."""
    return {
        "units_1": [64, 128, 256],
        "units_2": [32, 64, 128],
        "units_3": [16, 32, 64],
        "dropout_1": [0.2, 0.3, 0.4],
        "dropout_2": [0.2, 0.3, 0.4],
        "l2_reg": [1e-5, 1e-4, 1e-3],
        "learning_rate": [1e-4, 5e-4, 1e-3],
        "batch_size": [32, 64, 128],
        "epochs": [120, 200],
    }


def main():
    parser = argparse.ArgumentParser(description="Keras/TensorFlow NN — Grid Search Secuencial")
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
            name="KerasNN-sequential-gs",
            config={
                "model": "KerasBinaryNN",
                "param_grid": {k: [str(v) for v in vals] for k, vals in get_param_grid().items()},
                **reader.get_config(),
            },
        )

    estimator = KerasBinaryClassifier(random_state=args.seed, verbose=0)
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
    y_test_arr = np.asarray(y_test).reshape(-1)
    y_proba = gs.best_estimator_.predict_proba(X_test)[:, 1]    
    report = classification_report(y_test_arr, y_pred, output_dict=True)
    cm= confusion_matrix(y_test_arr, y_pred)
    auc = roc_auc_score(y_test_arr, y_proba)
    

    print("\n=== Evaluación en Test ===")
    print(classification_report(y_test_arr, y_pred))
    print("Matriz de confusión:")
    print(cm)
    print(f"AUC test: {auc:.4f}")
    print(f"Accuracy en test: {report['accuracy']:.4f}")
    print(f"Mejores hiperparámetros: {gs.best_params_}")

    gs.set_test_metrics({
        "test_accuracy": report["accuracy"],
        "test_f1_weighted": report["weighted avg"]["f1-score"],
        "test_precision_weighted": report["weighted avg"]["precision"],
        "test_recall_weighted": report["weighted avg"]["recall"],
        "test_roc_auc": auc,
    })

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_accuracy(gs.best_estimator_.history_, str(results_dir / "mlp_accuracy_curve.png"))

    if run is not None:
        run.log(
            {
                "test/accuracy": report["accuracy"],
                "test/f1_weighted": report["weighted avg"]["f1-score"],
                "test/precision_weighted": report["weighted avg"]["precision"],
                "test/recall_weighted": report["weighted avg"]["recall"],
                "test/roc_auc": auc,

            }
        )
        run.summary["best_params"] = gs.best_params_
        run.summary["best_val_score"] = gs.best_score_
        run.finish()

    gs.save_results("results/mlp")


if __name__ == "__main__":
    main()
