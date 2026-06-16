import argparse
import sys
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report , roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


from ml_utils import prepare_features



def build_model(model_name: str):
    model_name = model_name.lower()

    if model_name == "lr":
        return LogisticRegression(random_state=42 , max_iter=200)
    if model_name == "svm":
        return SVC(probability=True , random_state=42)
    if model_name == "rf":
        return RandomForestClassifier(random_state=42)
    if model_name == "xgb":
        return XGBClassifier(random_state=42 ,  eval_metric="logloss")

    raise ValueError("Modelo no soportado: usa lr|svm|rf|xgb")


def main():
    parser = argparse.ArgumentParser(description="Entrenar con un CSV y evaluar con data/test.csv")
    parser.add_argument("--csv", required=True, help="Ruta al CSV de entrenamiento")
    parser.add_argument("--model", required=True, choices=["lr", "svm", "rf", "xgb"],
                        help="Modelo: lr | svm | rf | xgb")
    args = parser.parse_args()

    # 1) Cargar TRAIN y preparar
    df_train = pd.read_csv(args.csv, keep_default_na=False)
    X_train, y_train = prepare_features(df_train)

    # 2) Cargar TEST fijo y preparar
    df_test = pd.read_csv("data/test.csv", keep_default_na=False)
    X_test, y_test = prepare_features(df_test)

    # 3) Alinear columnas (porque prepare_features hace one-hot y pueden salir columnas distintas)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # 4) Escalar SOLO si LR/SVM (porque son sensibles a escala)
    
    if args.model in ["lr", "svm"]:
        scaler = StandardScaler()

        X_train[["age", "pack_years"]] = scaler.fit_transform(
        X_train[["age", "pack_years"]]
        )

        X_test[["age", "pack_years"]] = scaler.transform(
        X_test[["age", "pack_years"]]
        )

    # 5) Entrenar modelo
    model = build_model(args.model)
    model.fit(X_train, y_train)

    # 6) Evaluar en TEST
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print("\n=== Evaluación en TEST (data/test.csv) ===")
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, y_proba)
    print("ROC-AUC:", roc_auc)
    print("\nReporte:")
    print(classification_report(y_test, y_pred, digits=4))

    


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n Error: {e}", file=sys.stderr)
        sys.exit(1)
