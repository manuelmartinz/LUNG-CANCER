import pandas as pd


df= pd.read_csv("data/ncr_thr95_k1.csv", keep_default_na=False)

print(df["lung_cancer"].value_counts())
print(df["lung_cancer"].value_counts(normalize=True))