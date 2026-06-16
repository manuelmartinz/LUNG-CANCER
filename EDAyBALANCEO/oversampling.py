import argparse
import os
from typing import List

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

TRAIN_PATH = "data/train.csv"
DEFAULT_OUTPUT_DIR = "data"
TARGET_COL = "lung_cancer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Oversampling con SMOTENC o RandomOverSampler sobre train.csv"
    )
    parser.add_argument(  
        "--algorithm",
        dest="algorithm",
        choices=["smotenc", "randomoversampler"],
        required=True,
        help="Algoritmo a usar: smotenc | randomoversampler",
    )
    parser.add_argument(
        "--proporcion_i",
        type=float,
        required=True,
        help="Proporción deseada de la clase minoritaria (p.ej. 0.4 para 60/40)",
    )
    return parser.parse_args()


def compute_sampling_strategy(target: pd.Series, desired_minority_prop: float) -> float:
    counts = target.value_counts()

    current_min_prop = counts.min() / counts.sum()
    if desired_minority_prop <= 0 or desired_minority_prop >= 1:
        raise ValueError("La proporción debe estar en (0, 1)")
    if desired_minority_prop < current_min_prop:
        raise ValueError(
            f"La proporción deseada ({desired_minority_prop:.3f}) es menor que la actual ({current_min_prop:.3f}); "
            "el oversampling no puede reducir la minoritaria"
        )

    # sampling_strategy es (minoritaria / mayoritaria) deseado
    return desired_minority_prop / (1.0 - desired_minority_prop)


def oversample_smote(df: pd.DataFrame, sampling_strategy: float) -> pd.DataFrame:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]


    if "patient_id" in X.columns:
        X = X.drop(columns=["patient_id"])

    num_cols: List[str] = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols: List[str] = X.select_dtypes(exclude=["number"]).columns.tolist()

    prep = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    X_enc = prep.fit_transform(X)

    # índices de categóricas en el array transformado: [num ... | cat ...]
    cat_idx = list(range(len(num_cols), len(num_cols) + len(cat_cols)))

    sampler = SMOTENC(
        categorical_features=cat_idx,
        sampling_strategy=sampling_strategy,
        random_state=42,
        k_neighbors=5,
    )
    X_res, y_res = sampler.fit_resample(X_enc, y)

    X_res = np.asarray(X_res)

    # revertir numéricas (desescalar)
    if num_cols:
        scaler: StandardScaler = prep.named_transformers_["num"]
        X_num = scaler.inverse_transform(X_res[:, : len(num_cols)])
        df_num = pd.DataFrame(X_num, columns=num_cols)

        if "age" in df_num.columns:
            df_num["age"] = df_num["age"].round().astype(int)
    else:
        df_num = pd.DataFrame(index=range(X_res.shape[0]))

    # revertir categóricas
    if cat_cols:
        enc: OrdinalEncoder = prep.named_transformers_["cat"]
        X_cat_codes = X_res[:, len(num_cols) :]

        # SMOTE puede generar valores intermedios: redondear a códigos enteros
        X_cat_codes = np.rint(X_cat_codes).astype(int)

        # recortar a rangos válidos por columna
        for j, cats in enumerate(enc.categories_):
            X_cat_codes[:, j] = np.clip(X_cat_codes[:, j], 0, len(cats) - 1)

        X_cat = enc.inverse_transform(X_cat_codes)
        df_cat = pd.DataFrame(X_cat, columns=cat_cols)
    else:
        df_cat = pd.DataFrame(index=range(X_res.shape[0]))

    df_res = pd.concat([df_num, df_cat], axis=1)
    df_res[TARGET_COL] = y_res.values

    # ordenar columnas como X + target (sin patient_id)
    df_res = df_res[X.columns.tolist() + [TARGET_COL]]
    return df_res


def oversample_random(df: pd.DataFrame, sampling_strategy: float) -> pd.DataFrame:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]


    sampler = RandomOverSampler(
        sampling_strategy=sampling_strategy,
        random_state=42,
    )
    X_res, y_res = sampler.fit_resample(X, y)

    df_res = pd.DataFrame(X_res, columns=X.columns)
    df_res[TARGET_COL] = y_res.values
    return df_res


def main() -> None:
    args = parse_args()

    df = pd.read_csv(TRAIN_PATH, keep_default_na=False)
    if TARGET_COL not in df.columns:
        raise ValueError(f"No se encontró la columna objetivo '{TARGET_COL}' en {TRAIN_PATH}")

    desired_prop = args.proporcion_i
    sampling_strategy = compute_sampling_strategy(df[TARGET_COL], desired_prop)

    if args.algorithm == "smotenc":
        df_res = oversample_smote(df, sampling_strategy)
    else:
        df_res = oversample_random(df, sampling_strategy)

    pct = int(round(desired_prop * 100))
    out_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{args.algorithm}_{pct}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_res.to_csv(out_path, index=False)

    orig_counts = df[TARGET_COL].value_counts()
    new_counts = df_res[TARGET_COL].value_counts()

    print("Distribución original:")
    print(orig_counts)
    print("\nDistribución tras oversampling:")
    print(new_counts)
    print(f"\nGuardado: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
