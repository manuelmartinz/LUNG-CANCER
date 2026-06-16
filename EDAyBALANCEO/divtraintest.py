import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------
# Configuración
# ----------------------------
CSV_PATH = "data/lung_cancer_dataset.csv"
TARGET_COL = "lung_cancer"
TEST_SIZE = 0.30
RANDOM_STATE = 42

# ----------------------------
# Cargar dataset
# ----------------------------
df = pd.read_csv(CSV_PATH, keep_default_na=False)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# ----------------------------
# Split estratificado
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,       # mantiene proporción de clases
    shuffle=True,
    random_state=RANDOM_STATE
)

# ----------------------------
# Reconstruir dataframes
# ----------------------------
train_df = X_train.copy()
train_df[TARGET_COL] = y_train.values

test_df = X_test.copy()
test_df[TARGET_COL] = y_test.values

# ----------------------------
# Guardar a CSV
# ----------------------------

train_df.to_csv("data//train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

# ----------------------------
# Comprobaciones
# ----------------------------
print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)

print("\nDistribución TRAIN:")
print(train_df[TARGET_COL].value_counts(normalize=True).round(3))

print("\nDistribución TEST:")
print(test_df[TARGET_COL].value_counts(normalize=True).round(3))
