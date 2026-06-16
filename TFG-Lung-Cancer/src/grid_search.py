"""
grid_search.py — Grid Search secuencial/incremental reutilizable.

Optimiza un hiperparámetro a la vez, actualizando el modelo
con el mejor valor encontrado antes de pasar al siguiente.
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold, cross_validate


class SequentialGridSearch:
    """
    Grid search incremental: optimiza un hiperparámetro por paso.

    Flujo:
        1. Partir del modelo con parámetros por defecto.
        2. Para cada hiperparámetro en orden:
            a. Probar todos los valores de su rejilla.
            b. Evaluar con StratifiedKFold.
            c. Escoger el mejor valor.
            d. Actualizar el modelo.
        3. Devolver el modelo final optimizado.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        param_grid: Dict[str, list],
        *,
        scoring: str = "f1_weighted",
        cv_splits: int = 5,
        random_state: int = 42,
        wandb_run: Optional[Any] = None,
    ):
        """
        Args:
            estimator: Estimador base de sklearn (o compatible).
            param_grid: Dict ordenado {nombre_param: [valores]}.
                        Se recorren en el orden de inserción.
            scoring: Métrica de sklearn para optimizar.
            cv_splits: Número de folds para StratifiedKFold.
            random_state: Semilla para reproducibilidad.
            wandb_run: Objeto wandb.run activo (opcional).
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.wandb_run = wandb_run

        # Resultados
        self.results_: List[dict] = []
        self.fold_results_: List[dict] = []
        self.best_params_: Dict[str, Any] = {}
        self.best_estimator_: Optional[BaseEstimator] = None
        self.best_score_: float = 0.0
        self.test_metrics_: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SequentialGridSearch":
        """
        Ejecuta el grid search secuencial.

        Args:
            X: Features de entrenamiento.
            y: Variable objetivo.

        Returns:
            self
        """
        skf = StratifiedKFold(
            n_splits=self.cv_splits, shuffle=True, random_state=self.random_state
        )

        current_params = {}
        step = 0
        # Recoge el mejor valor de cada paso para la tabla resumen final
        steps_summary: List[dict] = []

        for param_name, param_values in self.param_grid.items():
            step += 1
            best_val_score = -np.inf
            best_value = None
            found_valid_candidate = False
            # Resultados de este paso (uno por valor probado)
            step_records: List[dict] = []

            print(f"\n{'='*60}")
            print(f"Paso {step}: Optimizando '{param_name}' ({len(param_values)} valores)")
            print(f"{'='*60}")

            for value in param_values:
                trial_params = {**current_params, param_name: value}
                model = clone(self.estimator).set_params(**trial_params)

                t0 = time.time()
                try:
                    cv_results = cross_validate(
                        model, X, y,
                        cv=skf,
                        scoring={
                            "accuracy": "accuracy",
                            "f1_weighted": "f1_weighted",
                            "precision_weighted": "precision_weighted",
                            "recall_weighted": "recall_weighted",
                        },
                        return_train_score=True,
                        n_jobs=-1,
                    )
                except Exception as exc:
                    elapsed = time.time() - t0
                    value_str = str(value)
                    iteration_record = {
                        "step": step,
                        "param_name": param_name,
                        "param_value": value_str,
                        "train_score_mean": np.nan,
                        "train_score_std": np.nan,
                        "val_score_mean": np.nan,
                        "val_score_std": np.nan,
                        "val_accuracy_mean": np.nan,
                        "val_accuracy_std": np.nan,
                        "val_f1_mean": np.nan,
                        "val_f1_std": np.nan,
                        "val_precision_mean": np.nan,
                        "val_precision_std": np.nan,
                        "val_recall_mean": np.nan,
                        "val_recall_std": np.nan,
                        "elapsed_seconds": elapsed,
                        "status": "failed",
                        "error": str(exc),
                    }
                    self.results_.append(iteration_record)
                    step_records.append(iteration_record)
                    print(
                        f"  {param_name}={value_str:<20} | ERROR: {exc}"
                    )
                    continue

                elapsed = time.time() - t0

                # Métricas de validación
                val_score = cv_results[f"test_{self.scoring}"].mean()
                val_std = cv_results[f"test_{self.scoring}"].std()
                train_score = cv_results[f"train_{self.scoring}"].mean()
                train_std = cv_results[f"train_{self.scoring}"].std()

                # Registro por fold
                for fold_i in range(self.cv_splits):
                    fold_record = {
                        "step": step,
                        "param_name": param_name,
                        "param_value": str(value),
                        "fold": fold_i + 1,
                        "train_accuracy": cv_results["train_accuracy"][fold_i],
                        "val_accuracy": cv_results["test_accuracy"][fold_i],
                        "train_f1": cv_results["train_f1_weighted"][fold_i],
                        "val_f1": cv_results["test_f1_weighted"][fold_i],
                        "train_precision": cv_results["train_precision_weighted"][fold_i],
                        "val_precision": cv_results["test_precision_weighted"][fold_i],
                        "train_recall": cv_results["train_recall_weighted"][fold_i],
                        "val_recall": cv_results["test_recall_weighted"][fold_i],
                    }
                    self.fold_results_.append(fold_record)

                # Registro por iteración
                iteration_record = {
                    "step": step,
                    "param_name": param_name,
                    "param_value": str(value),
                    "train_score_mean": train_score,
                    "train_score_std": train_std,
                    "val_score_mean": val_score,
                    "val_score_std": val_std,
                    "val_accuracy_mean": cv_results["test_accuracy"].mean(),
                    "val_accuracy_std": cv_results["test_accuracy"].std(),
                    "val_f1_mean": cv_results["test_f1_weighted"].mean(),
                    "val_f1_std": cv_results["test_f1_weighted"].std(),
                    "val_precision_mean": cv_results["test_precision_weighted"].mean(),
                    "val_precision_std": cv_results["test_precision_weighted"].std(),
                    "val_recall_mean": cv_results["test_recall_weighted"].mean(),
                    "val_recall_std": cv_results["test_recall_weighted"].std(),
                    "elapsed_seconds": elapsed,
                    "status": "ok",
                    "error": "",
                }
                self.results_.append(iteration_record)
                step_records.append(iteration_record)

                status = ""
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_value = value
                    found_valid_candidate = True
                    status = " ★ BEST"

                print(
                    f"  {param_name}={str(value):<20} | "
                    f"val_{self.scoring}={val_score:.4f}±{val_std:.4f} | "
                    f"train={train_score:.4f}±{train_std:.4f} | "
                    f"{elapsed:.1f}s{status}"
                )

            if not found_valid_candidate:
                raise ValueError(
                    f"No hubo configuraciones válidas para '{param_name}' "
                    f"con parámetros actuales: {current_params}"
                )

            # Actualizar con el mejor valor encontrado
            current_params[param_name] = best_value
            steps_summary.append({
                "step": step,
                "param_name": param_name,
                "best_value": str(best_value),
                "best_val_score": best_val_score,
            })
            print(f"\n  ► Mejor valor para '{param_name}': {best_value} "
                  f"(val_{self.scoring}={best_val_score:.4f})")

            # ── Logging a wandb (al final de cada paso) ──────────────────
            if self.wandb_run is not None:
                self._log_step_wandb(step, param_name, step_records, best_value, best_val_score)

        # Modelo final
        self.best_params_ = current_params
        self.best_estimator_ = clone(self.estimator).set_params(**current_params)
        self.best_estimator_.fit(X, y)
        self.best_score_ = best_val_score

        print(f"\n{'='*60}")
        print(f"Optimización finalizada")
        print(f"Mejores hiperparámetros: {self.best_params_}")
        print(f"Mejor val_{self.scoring}: {self.best_score_:.4f}")
        print(f"{'='*60}")

        # Tabla resumen global en wandb
        if self.wandb_run is not None:
            self._log_summary_wandb(steps_summary)

        return self

    # ── Métodos de logging wandb ─────────────────────────────────────────

    def _log_step_wandb(
        self,
        step: int,
        param_name: str,
        step_records: List[dict],
        best_value: Any,
        best_val_score: float,
    ) -> None:
        """
        Registra en wandb, para un paso completo:
          - Tabla con todos los valores probados y sus métricas.
          - Gráfica de barras: val_f1 por valor de hiperparámetro.
          - Gráfica de líneas: train_f1 vs val_f1 para cada valor.
        """
        import wandb as _wandb

        labels = [r["param_value"] for r in step_records]
        val_f1 = [r["val_f1_mean"] for r in step_records]
        val_f1_std = [r["val_f1_std"] for r in step_records]
        train_f1 = [r["train_score_mean"] for r in step_records]
        val_acc = [r["val_accuracy_mean"] for r in step_records]
        val_prec = [r["val_precision_mean"] for r in step_records]
        val_rec = [r["val_recall_mean"] for r in step_records]

        # ── Tabla ────────────────────────────────────────────────────────
        table = _wandb.Table(
            columns=[
                "param_value", "val_f1_mean", "val_f1_std",
                "train_f1_mean", "val_accuracy", "val_precision", "val_recall",
            ],
            data=[
                [labels[i], val_f1[i], val_f1_std[i], train_f1[i],
                 val_acc[i], val_prec[i], val_rec[i]]
                for i in range(len(labels))
            ],
        )

        # ── Gráfica de barras: val_f1 por valor ──────────────────────────
        bar_chart = _wandb.plot.bar(
            table,
            label="param_value",
            value="val_f1_mean",
            title=f"Paso {step} — {param_name}: val F1 por valor",
        )

        # ── Gráfica de líneas: train vs val F1 ───────────────────────────
        xs = list(range(len(labels)))
        line_chart = _wandb.plot.line_series(
            xs=xs,
            ys=[val_f1, train_f1],
            keys=["val_f1", "train_f1"],
            title=f"Paso {step} — {param_name}: Train vs Val F1",
            xname="índice valor",
        )

        self.wandb_run.log({
            f"step_{step}/tabla_{param_name}": table,
            f"step_{step}/barras_val_f1": bar_chart,
            f"step_{step}/lineas_train_vs_val": line_chart,
            f"step_{step}/mejor_valor": str(best_value),
            f"step_{step}/mejor_val_f1": best_val_score,
        })

    def _log_summary_wandb(self, steps_summary: List[dict]) -> None:
        """
        Registra en wandb una tabla resumen con el mejor valor
        encontrado en cada paso y su puntuación.
        """
        import wandb as _wandb

        summary_table = _wandb.Table(
            columns=["step", "param_name", "best_value", "best_val_score"],
            data=[
                [r["step"], r["param_name"], r["best_value"], r["best_val_score"]]
                for r in steps_summary
            ],
        )
        self.wandb_run.log({"resumen_optimizacion": summary_table})
        self.wandb_run.summary["best_params"] = self.best_params_
        self.wandb_run.summary["best_val_score"] = self.best_score_

    def get_results_df(self) -> pd.DataFrame:
        """Devuelve un DataFrame con los resultados por iteración."""
        return pd.DataFrame(self.results_)

    def get_fold_results_df(self) -> pd.DataFrame:
        """Devuelve un DataFrame con los resultados por fold."""
        return pd.DataFrame(self.fold_results_)

    def get_summary_df(self) -> pd.DataFrame:
        """
        Devuelve un DataFrame resumen con medias y desviaciones
        agrupadas por paso e hiperparámetro.
        """
        df = self.get_fold_results_df()
        if df.empty:
            return df

        metrics = [c for c in df.columns if c.startswith(("train_", "val_"))]
        grouped = df.groupby(["step", "param_name", "param_value"])[metrics]

        summary = grouped.agg(["mean", "std"]).reset_index()
        summary.columns = [
            f"{a}_{b}" if b else a for a, b in summary.columns
        ]

        # Adjuntar una fila final con las métricas de test del mejor modelo.
        if self.test_metrics_:
            test_row = {col: np.nan for col in summary.columns}
            test_row["step"] = "test"
            test_row["param_name"] = "best_params"
            test_row["param_value"] = "best_estimator"
            for key, value in self.test_metrics_.items():
                if key in summary.columns:
                    test_row[key] = value
                else:
                    summary[key] = np.nan
                    test_row[key] = value

            summary = pd.concat([summary, pd.DataFrame([test_row])], ignore_index=True)

        return summary

    def set_test_metrics(self, metrics: Dict[str, Any]) -> None:
        """Registra métricas de test para guardarlas también en el summary."""
        clean_metrics: Dict[str, Any] = {}
        for key, value in metrics.items():
            if value is None:
                clean_metrics[key] = np.nan
            elif isinstance(value, (np.integer, np.floating)):
                clean_metrics[key] = float(value)
            else:
                clean_metrics[key] = value
        self.test_metrics_ = clean_metrics

    def save_results(self, prefix: str = "models/results/output") -> None:
        """
        Guarda todos los resultados en CSV:
          - {prefix}_iterations.csv
          - {prefix}_folds.csv
          - {prefix}_summary.csv
          - {prefix}_best_params.csv
        """
        prefix_path = Path(prefix)
        prefix_path.parent.mkdir(parents=True, exist_ok=True)
        prefix_str = str(prefix_path)

        best_params_df = pd.DataFrame([
            {
                key: "None" if value is None else value
                for key, value in self.best_params_.items()
            }
        ])

        self.get_results_df().to_csv(f"{prefix_str}_iterations.csv", index=False)
        self.get_fold_results_df().to_csv(f"{prefix_str}_folds.csv", index=False)
        self.get_summary_df().to_csv(f"{prefix_str}_summary.csv", index=False)
        best_params_df.to_csv(
            f"{prefix_str}_best_params.csv", index=False
        )
        print(f"Resultados guardados con prefijo: {prefix_str}")
