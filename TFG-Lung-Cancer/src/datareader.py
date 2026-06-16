"""
datareader.py — Módulo de carga y preprocesamiento de datos.

Soporta variables categóricas y numéricas con imputación,
codificación, escalado y transformaciones configurables.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer


class DataReader:
    """Carga un CSV y aplica preprocesamiento configurable."""

    def __init__(
        self,
        filepath: str,
        target_col: str = "lung_cancer",
        drop_cols: Optional[list] = ["patient_id"],
        n_rows: Optional[int] = None,
        encoding: str = "onehot",
        scale: bool = True,
        impute_strategy_num: str = "median",
        impute_strategy_cat: str = "most_frequent",
        random_state: int = 42,
    ):
        """
        Args:
            filepath: Ruta al archivo CSV.
            target_col: Nombre de la columna objetivo.
            drop_cols: Columnas a eliminar antes de procesar.
            n_rows: Si se indica, toma un subconjunto aleatorio de n filas.
            encoding: Tipo de codificación para categóricas ('onehot' | 'ordinal' | 'label' | 'none' | 'catboost').
            scale: Si True, aplica StandardScaler a las variables numéricas.
            impute_strategy_num: Estrategia de imputación para numéricas.
            impute_strategy_cat: Estrategia de imputación para categóricas.
            random_state: Semilla para reproducibilidad.
        """
        self.filepath = filepath
        self.target_col = target_col
        self.drop_cols = drop_cols or ["patient_id"]
        self.n_rows = n_rows
        self.encoding = encoding
        self.scale = scale
        self.impute_strategy_num = impute_strategy_num
        self.impute_strategy_cat = impute_strategy_cat
        self.random_state = random_state

        self._scaler: Optional[StandardScaler] = None
        self._num_imputer: Optional[SimpleImputer] = None
        self._cat_imputer: Optional[SimpleImputer] = None
        self._ordinal_encoder: Optional[OrdinalEncoder] = None
        self._label_encoders: dict = {}

    def _resolve_filepath(self) -> Path:
        """Resuelve rutas comunes del dataset para evitar errores por ruta incompleta."""
        raw = Path(self.filepath)
        candidates = []

        # 1) Ruta exacta tal cual llega por CLI/config
        candidates.append(raw)

        # 2) Misma ruta con extensión .csv si no se indicó
        if raw.suffix == "":
            candidates.append(raw.with_suffix(".csv"))

        # 3) Ubicación habitual del proyecto: src/data/
        project_root = Path(__file__).resolve().parent.parent
        data_dir = project_root / "src" / "data"
        candidates.append(data_dir / raw.name)
        if raw.suffix == "":
            candidates.append(data_dir / f"{raw.name}.csv")

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate

        attempted = "\n".join(f"- {c}" for c in candidates)
        raise FileNotFoundError(
            "No se encontró el dataset. Rutas intentadas:\n"
            f"{attempted}\n"
            "Sugerencia: usa --data src/data/lung_cancer_dataset.csv"
        )

    def load(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Carga y preprocesa el dataset completo.

        Returns:
            Tupla (X, y) con las features procesadas y la variable objetivo.
        """
        resolved_path = self._resolve_filepath()
        df = pd.read_csv(resolved_path, keep_default_na=False)

        if self.n_rows is not None:
            df = df.sample(n=self.n_rows, random_state=self.random_state)

        # Eliminar columnas no deseadas
        for col in self.drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Separar objetivo
        y = self._encode_target(df[self.target_col])
        X = df.drop(columns=[self.target_col])

        # Identificar tipos
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X.select_dtypes(include=["number"]).columns.tolist()

        # Imputación numérica
        if num_cols:
            self._num_imputer = SimpleImputer(strategy=self.impute_strategy_num)
            X[num_cols] = self._num_imputer.fit_transform(X[num_cols])

        # Imputación categórica
        if cat_cols:
            self._cat_imputer = SimpleImputer(strategy=self.impute_strategy_cat)
            X[cat_cols] = self._cat_imputer.fit_transform(X[cat_cols])

        # Codificación categórica
        X = self._encode_categoricals(X, cat_cols)

        # Escalado
        if self.scale:
            num_cols_final = X.select_dtypes(include=["number"]).columns.tolist()
            if num_cols_final:
                self._scaler = StandardScaler()
                X[num_cols_final] = self._scaler.fit_transform(X[num_cols_final])

        return X, y

    def load_split(
        self, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Carga, preprocesa y divide en train/test estratificado.

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        X, y = self.load()
        return train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=self.random_state,
        )

    def _encode_target(self, series: pd.Series) -> pd.Series:
        """Convierte la columna objetivo a valores numéricos si es necesario."""
        print(f"Series dtype: {series.dtype}")
        if series.dtype == "string" or series.dtype == "object":
            mapping = {v: i for i, v in enumerate(sorted(series.unique()))}
            return series.map(mapping)
        return series

    def _encode_categoricals(
        self, X: pd.DataFrame, cat_cols: list
    ) -> pd.DataFrame:
        """Aplica la codificación configurada a las columnas categóricas."""
        if not cat_cols:
            return X

        if self.encoding in ("none", "catboost"):
            return X

        if self.encoding == "onehot":
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
            # Asegurar tipos numéricos (get_dummies genera bool en nuevas versiones)
            bool_cols = X.select_dtypes(include=["bool"]).columns
            X[bool_cols] = X[bool_cols].astype(int)

        elif self.encoding == "ordinal":
            self._ordinal_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            X[cat_cols] = self._ordinal_encoder.fit_transform(X[cat_cols])

        elif self.encoding == "label":
            for col in cat_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self._label_encoders[col] = le

        return X

    def get_config(self) -> dict:
        """Devuelve la configuración del DataReader como diccionario."""
        return {
            "filepath": self.filepath,
            "target_col": self.target_col,
            "drop_cols": self.drop_cols,
            "n_rows": self.n_rows,
            "encoding": self.encoding,
            "scale": self.scale,
            "impute_strategy_num": self.impute_strategy_num,
            "impute_strategy_cat": self.impute_strategy_cat,
            "random_state": self.random_state,
        }