import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import mutual_info_classif



pd.set_option("display.max_columns", None)

df=pd.read_csv('data/lung_cancer_dataset.csv' ,keep_default_na=False)


print(df.head())
print(df.shape)
print(df.dtypes)
print(df.describe(include="all").T)

# NaN reales 
print("NaN reales:", df["alcohol_consumption"].isna().sum())

# Contar la categoría "None" 
print('"None" (texto):', (df["alcohol_consumption"] == "None").sum())

# Ver todas las categorías y sus frecuencias
print(df["alcohol_consumption"].value_counts(dropna=False))

print(df.duplicated().sum())
print(df["patient_id"].nunique(), len(df))
for c in ["gender","radon_exposure","asbestos_exposure","secondhand_smoke_exposure",
          "copd_diagnosis","alcohol_consumption","family_history","lung_cancer"]:
    print(c, sorted(df[c].unique()))


print(df[["age","pack_years"]].agg(["min","max"]))


target = "lung_cancer"
id_col = "patient_id"
num_cols = ["age", "pack_years"]
cat_cols = [c for c in df.columns if c not in [id_col, target] + num_cols]


vc = df[target].value_counts()
print(vc)
print((vc/len(df)*100).round(2))

baseline = vc.max()/len(df)
print("Baseline accuracy (clase mayoritaria):", round(baseline, 4))


vc.plot(kind="bar")
plt.title("Distribución de lung_cancer")
plt.xlabel("Clase")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("lung_cancer_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

print(df[["age","pack_years"]].describe().T)


#Distribucion de las variables numéricas

for c in ["age", "pack_years"]:
    df[c].hist(bins=40)
    plt.xlabel(c, fontsize=14)
    plt.ylabel("Frecuencia", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"histogram_{c}.png", dpi=300, bbox_inches="tight")
    plt.close()


counts = df["age"].value_counts().sort_index()
print(counts.head(10))
print(counts.tail(10))
print(counts.describe())



#Distribucion de las variables categóricas

cat_cols = ["gender","radon_exposure","asbestos_exposure","secondhand_smoke_exposure",
            "copd_diagnosis","alcohol_consumption","family_history"]

for c in cat_cols:
    vc = df[c].value_counts()
    print("\n", c)
    print(vc)
    print((vc/len(df)*100).round(2))


cols_plot = 2
rows_plot = int(np.ceil(len(cat_cols) / cols_plot))
fig, axes = plt.subplots(rows_plot, cols_plot, figsize=(14, 4 * rows_plot))
axes = np.array(axes).reshape(-1)

for ax, c in zip(axes, cat_cols):
    vc = df[c].value_counts()
    percentages = (vc / len(df) * 100).round(1)
    bars = ax.bar(vc.index.astype(str), vc.values, width=0.55, color="#4C72B0", edgecolor="black", linewidth=0.6)
    ax.set_title(c.replace("_", " ").title(), fontsize=14)
    ax.set_ylabel("Frecuencia", fontsize=13)
    ax.tick_params(axis="x", labelrotation=35, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    if len(vc) > 0:
        ax.set_ylim(0, vc.values.max() * 1.22)

    for bar, pct in zip(bars, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{pct}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )

for ax in axes[len(cat_cols):]:
    ax.axis("off")

fig.suptitle("Distribución de las variables categóricas", fontsize=18, y=1.02)
fig.tight_layout()
fig.savefig("categorical_distributions.png", dpi=300, bbox_inches="tight")
plt.close(fig)



#Análisis de tasa de cáncer condicionada a la categoría.



for c in cat_cols:
    tab = pd.crosstab(df[c], df["lung_cancer"], normalize="index")
    tab.columns = ["No_rate", "Yes_rate"]
    print("\n", c)
    print(tab.round(3))



print(df.groupby("lung_cancer")[["age","pack_years"]].agg(["mean","median","std","min","max"]))

for c in ["age", "pack_years"]:
    df.boxplot(column=c, by="lung_cancer")
    plt.title(f"{c} por lung_cancer")
    plt.suptitle("")
    plt.xlabel("lung_cancer")
    plt.ylabel(c)
    plt.tight_layout()
    plt.savefig(f"boxplot_{c}_by_lung_cancer.png", dpi=300, bbox_inches="tight")
    plt.close()




# Crear etiqueta numérica
df["y"] = (df["lung_cancer"] == "Yes").astype(int)

cols_num_y = ["age", "pack_years", "y"]

# =========================
# Pearson (age, pack_years, y)
# =========================
corrpe = df[cols_num_y].corr(method="pearson")
print("\nPearson corr (con y):\n", corrpe)

plt.figure(figsize=(5, 4))
plt.imshow(corrpe.values, vmin=-1, vmax=1)
plt.title("Heatmap Pearson (numéricas + y)")
plt.xticks(range(len(cols_num_y)), cols_num_y, rotation=45, ha="right")
plt.yticks(range(len(cols_num_y)), cols_num_y)
plt.colorbar(label="Pearson r")

for i in range(len(cols_num_y)):
    for j in range(len(cols_num_y)):
        plt.text(j, i, f"{corrpe.values[i, j]:.2f}",
                 ha="center", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("heatmap_pearson_numericas_y.png", dpi=300, bbox_inches="tight")
plt.close()
print("Guardado: heatmap_pearson_numericas_y.png")


# =========================
# Spearman (age, pack_years, y)
# =========================
corrsp = df[cols_num_y].corr(method="spearman")
print("\nSpearman corr (con y):\n", corrsp)

plt.figure(figsize=(5, 4))
plt.imshow(corrsp.values, vmin=-1, vmax=1)
plt.title("Heatmap Spearman (numéricas + y)")
plt.xticks(range(len(cols_num_y)), cols_num_y, rotation=45, ha="right")
plt.yticks(range(len(cols_num_y)), cols_num_y)
plt.colorbar(label="Spearman ρ")

for i in range(len(cols_num_y)):
    for j in range(len(cols_num_y)):
        plt.text(j, i, f"{corrsp.values[i, j]:.2f}",
                 ha="center", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("heatmap_spearman_numericas_y.png", dpi=300, bbox_inches="tight")
plt.close()
print("Guardado: heatmap_spearman_numericas_y.png")




#Correlacion variables categoricas


def cramers_v(x, y):
    tab = pd.crosstab(x, y)
    chi2 = chi2_contingency(tab)[0]
    n = tab.values.sum()
    r, k = tab.shape
    return np.sqrt((chi2/n) / (min(r-1, k-1)))

target = "lung_cancer"
cat_cols = ["gender","radon_exposure","asbestos_exposure","secondhand_smoke_exposure",
            "copd_diagnosis","alcohol_consumption","family_history"]

cols = cat_cols + [target]
M = pd.DataFrame(index=cols, columns=cols, dtype=float)

for a in cols:
    for b in cols:
        if a == b:
            M.loc[a, b] = 1.0
        else:
            M.loc[a, b] = cramers_v(df[a], df[b])

plt.figure(figsize=(9, 7))
plt.imshow(M.values, vmin=0, vmax=1, cmap="Blues")   # heatmap simple sin seaborn
plt.title("Heatmap de Cramér's V (categóricas + etiqueta)")
plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
plt.yticks(range(len(cols)), cols)
plt.colorbar(label="Cramér's V")

# escribir valores dentro (opcional)
for i in range(len(cols)):
    for j in range(len(cols)):
        plt.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=8)

plt.tight_layout()
plt.savefig("correlacionvarcat.png", dpi=300, bbox_inches="tight")
plt.close()


X = df.drop(columns=[target, "patient_id", "y"], errors="ignore")
y = (df[target] == "Yes").astype(int)


assert "y" not in X.columns
assert target not in X.columns

# Detectar columnas numéricas y categóricas sin renombrar nada
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# ----------------------------
# 1) Information Value (IV)
# ----------------------------
def iv_from_series(x, y, eps=0.5):
    tab = pd.crosstab(x, y)

    if 0 not in tab.columns:
        tab[0] = 0
    if 1 not in tab.columns:
        tab[1] = 0

    tab = tab[[0, 1]].astype(float)
    tab[0] += eps
    tab[1] += eps

    dist0 = tab[0] / tab[0].sum()   # distribución en clase 0
    dist1 = tab[1] / tab[1].sum()   # distribución en clase 1

    woe = np.log(dist1 / dist0)
    iv = ((dist1 - dist0) * woe).sum()
    return float(iv)

def compute_iv(X, y, num_cols, cat_cols, bins=10, eps=0.5):
    ivs = {}

    # Categóricas: tal cual (NaN como categoría)
    for c in cat_cols:
        ivs[c] = iv_from_series(X[c].fillna("__MISSING__"), y, eps=eps)

    # Numéricas: binning en cuantiles (qcut) + NaN como categoría
    for c in num_cols:
        s = X[c]
        b = pd.qcut(s, q=bins, duplicates="drop")  # crea bins tipo (a,b]
        b = b.astype(object)
        b = b.where(~s.isna(), "__MISSING__")
        ivs[c] = iv_from_series(b, y, eps=eps)

    return pd.Series(ivs).sort_values(ascending=False)

iv_series = compute_iv(X, y, num_cols, cat_cols, bins=10, eps=0.5)
print("\n=== IV por variable ===")
print(iv_series.round(4))

# ----------------------------
# 2) Mutual Information (MI)
# ----------------------------
pre = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

X_enc = pre.fit_transform(X)
mi = mutual_info_classif(X_enc, y, random_state=42)

feature_names = pre.get_feature_names_out()
mi_df = pd.DataFrame({"feature": feature_names, "mi": mi})

# Agregar MI por variable original (robusto con guiones bajos)
def base_name(f, cat_cols):
    if f.startswith("num__"):
        return f.split("__", 1)[1]

    if f.startswith("cat__"):
        rest = f.split("__", 1)[1]  # ej: "secondhand_smoke_exposure_Yes"
        for c in sorted(cat_cols, key=len, reverse=True):
            if rest == c or rest.startswith(c + "_"):
                return c
        return rest.rsplit("_", 1)[0]  # fallback

    return f

mi_df["base"] = mi_df["feature"].apply(lambda f: base_name(f, cat_cols))
mi_series = mi_df.groupby("base")["mi"].sum().sort_values(ascending=False)

print("\n=== MI por variable ===")
print(mi_series.round(4))

# ----------------------------
# 3) Gráficas
# ----------------------------
def plot_barh(series, title, xlabel, filename):
    s = series.sort_values(ascending=True)
    plt.figure(figsize=(9, 5))
    plt.barh(s.index.astype(str), s.values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

plot_barh(iv_series, "Information Value (IV) por variable", "IV", "iv_por_variable.png")
plot_barh(mi_series, "Información Mutua (MI) por variable", "MI", "mi_por_variable.png")