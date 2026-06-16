import argparse
import os
from typing import List

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler , NeighbourhoodCleaningRule
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

TRAIN_PATH = "data/train.csv"
DEFAULT_OUTPUT_DIR = "data"
TARGET_COL = "lung_cancer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Undersampling con RandomUnderSampler y NCR sobre train.csv"
    )
    parser.add_argument(  
        "--algorithm",
        dest="algorithm",
        choices=["rus", "ncr"],
        required=True,
        help="Algoritmo a usar: rus | ncr",
    )
    parser.add_argument(
        "--proporcion_i",
        type=float,
        required=False,
        help="Proporción deseada de la clase minoritaria (p.ej. 0.4 para 60/40) , solo para RandomUnderSampler",
    )
    parser.add_argument(

        "--thresholdcleaning",
        type=float,
        required=False,
        help="Umbral para el algoritmo NeighbourhoodCleaningRule , solo para NCR",
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        required=False,
        help="Número de vecinos para NCR , solo para NCR",


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
            "el undersampling de la mayoría no puede reducir aún más la minoritaria"
        )

    # sampling_strategy es (minoritaria / mayoritaria) deseado
    return desired_minority_prop / (1.0 - desired_minority_prop)


def undersample_ncr(df: pd.DataFrame , thres_cle: float , neighbors: int) -> pd.DataFrame:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].map({"No": 0, "Yes": 1})

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
    sampler = NeighbourhoodCleaningRule(
        n_neighbors=neighbors,
        threshold_cleaning=thres_cle,
        
    )
    
    sampler.fit_resample(X_enc, y)

    if not hasattr(sampler, "sample_indices_"):
        raise AttributeError(
            "Tu NeighbourhoodCleaningRule no expone sample_indices_. "
            "Sin eso, no puedo devolverte el DF original exacto; habría que reconstruir manualmente."
        )

    idx = sampler.sample_indices_
    return df.iloc[idx].copy().reset_index(drop=True)


def undersample_random(df: pd.DataFrame, sampling_strategy: float) -> pd.DataFrame:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]


    sampler = RandomUnderSampler(
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
    

    if args.algorithm == "rus":

        if args.proporcion_i is None:
            raise ValueError("--proporcion_i es obligatorio cuando --algorithm rus")

        desired_prop = args.proporcion_i
        sampling_strategy = compute_sampling_strategy(df[TARGET_COL], desired_prop)
        df_res = undersample_random(df, sampling_strategy)

        pct = int(round(desired_prop * 100))
        out_path = os.path.join(DEFAULT_OUTPUT_DIR, f"rus_{pct}.csv")

    else: 

        if args.thresholdcleaning is None:
            raise ValueError("--thresholdcleaning es obligatorio cuando --algorithm ncr")

        if args.n_neighbors is None:
            raise ValueError("--n_neighbors es obligatorio cuando --algorithm ncr")


        if args.n_neighbors < 1:
            raise ValueError("--n_neighbors debe ser >= 1")    

        if not (0 < args.thresholdcleaning < 1):
            raise ValueError("--thresholdcleaning debe estar en (0, 1)")

        df_res = undersample_ncr(df, args.thresholdcleaning , args.n_neighbors)

        thr = int(round(args.thresholdcleaning * 100))
        out_path = os.path.join(DEFAULT_OUTPUT_DIR, f"ncr_thr{thr}_k{args.n_neighbors}.csv")



    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_res.to_csv(out_path, index=False)

    orig_counts = df[TARGET_COL].value_counts()
    new_counts = df_res[TARGET_COL].value_counts()

    print("Distribución original:")
    print(orig_counts)
    print("\nDistribución tras undersampling:")
    print(new_counts)
    print(f"\nGuardado: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
