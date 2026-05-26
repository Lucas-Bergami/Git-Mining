import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURAÇÕES
# =========================================================

INPUT_FILE = "data/processed/user_repo_dataset.csv"

OUTPUT_DIR = "analysis"

HIST_DIR = f"{OUTPUT_DIR}/histograms"
BOX_DIR = f"{OUTPUT_DIR}/boxplots"
CORR_DIR = f"{OUTPUT_DIR}/correlation"
LABEL_DIR = f"{OUTPUT_DIR}/label_analysis"

# =========================================================
# PASTAS
# =========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(HIST_DIR, exist_ok=True)
os.makedirs(BOX_DIR, exist_ok=True)
os.makedirs(CORR_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# =========================================================
# CARREGAR DATASET
# =========================================================

print("\nCarregando dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Shape: {df.shape}")

# =========================================================
# REMOVER COLUNAS NÃO NUMÉRICAS
# =========================================================

ignore_columns = [
    "username",
    "repo_name",
]

numeric_df = df.drop(columns=ignore_columns)

# =========================================================
# IDENTIFICAR COLUNAS NUMÉRICAS
# =========================================================

numeric_columns = numeric_df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print(f"\nColunas numéricas: {len(numeric_columns)}")

# =========================================================
# ESTATÍSTICAS BÁSICAS
# =========================================================

print("\nGerando estatísticas básicas...")

stats = []

for col in numeric_columns:
    stats.append(
        {
            "feature": col,
            "min": numeric_df[col].min(),
            "max": numeric_df[col].max(),
            "mean": numeric_df[col].mean(),
            "median": numeric_df[col].median(),
            "std": numeric_df[col].std(),
        }
    )

stats_df = pd.DataFrame(stats)

stats_df.to_csv(f"{OUTPUT_DIR}/statistics.csv", index=False)

print("Arquivo salvo:")
print(f"{OUTPUT_DIR}/statistics.csv")

# =========================================================
# HISTOGRAMAS
# =========================================================

print("\nGerando histogramas...")

for col in numeric_columns:
    plt.figure(figsize=(8, 5))

    plt.hist(numeric_df[col], bins=30)

    plt.title(f"Histogram - {col}")

    plt.xlabel(col)

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(f"{HIST_DIR}/{col}_hist.png")

    plt.close()

print("Histogramas salvos.")

# =========================================================
# BOXPLOTS
# =========================================================

print("\nGerando boxplots...")

for col in numeric_columns:
    plt.figure(figsize=(8, 5))

    plt.boxplot(numeric_df[col], vert=False)

    plt.title(f"Boxplot - {col}")

    plt.xlabel(col)

    plt.tight_layout()

    plt.savefig(f"{BOX_DIR}/{col}_boxplot.png")

    plt.close()

print("Boxplots salvos.")

# =========================================================
# MATRIZ DE CORRELAÇÃO
# =========================================================

print("\nGerando matriz de correlação...")

corr_matrix = numeric_df.corr(numeric_only=True)

corr_matrix.to_csv(f"{CORR_DIR}/correlation_matrix.csv")

# =========================================================
# HEATMAP
# =========================================================

plt.figure(figsize=(18, 14))

plt.imshow(corr_matrix, aspect="auto")

plt.colorbar()

plt.xticks(
    range(len(corr_matrix.columns)),
    corr_matrix.columns,
    rotation=90,
)

plt.yticks(
    range(len(corr_matrix.columns)),
    corr_matrix.columns,
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig(f"{CORR_DIR}/correlation_heatmap.png")

plt.close()

print("Correlação salva.")

# =========================================================
# ANÁLISE POR LABEL
# =========================================================

print("\nGerando análise por classe...")

label_0 = df[df["label"] == 0]
label_1 = df[df["label"] == 1]

for col in numeric_columns:
    if col == "label":
        continue

    plt.figure(figsize=(8, 5))

    plt.hist(label_0[col], bins=30, alpha=0.5, label="Label 0")

    plt.hist(label_1[col], bins=30, alpha=0.5, label="Label 1")

    plt.title(f"Label Distribution - {col}")

    plt.xlabel(col)

    plt.ylabel("Frequency")

    plt.legend()

    plt.tight_layout()

    plt.savefig(f"{LABEL_DIR}/{col}_label_dist.png")

    plt.close()

print("Análises por label salvas.")

# =========================================================
# TOP CORRELAÇÕES COM LABEL
# =========================================================

print("\nGerando ranking de features...")

label_corr = corr_matrix["label"].sort_values(ascending=False)

label_corr.to_csv(f"{CORR_DIR}/label_correlations.csv")

# =========================================================
# FEATURE IMPORTANCE VISUAL
# =========================================================

top_corr = label_corr.drop("label")

plt.figure(figsize=(10, 8))

plt.barh(top_corr.index, top_corr.values)

plt.title("Feature Correlation with Label")

plt.xlabel("Correlation")

plt.tight_layout()

plt.savefig(f"{CORR_DIR}/label_correlations.png")

plt.close()

# =========================================================
# DISTRIBUIÇÃO DAS CLASSES
# =========================================================

plt.figure(figsize=(6, 5))

df["label"].value_counts().plot(kind="bar")

plt.title("Class Distribution")

plt.xlabel("Label")

plt.ylabel("Count")

plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/class_distribution.png")

plt.close()

# =========================================================
# FINAL
# =========================================================

print("\n======================================")
print("ANÁLISE EXPLORATÓRIA FINALIZADA")
print("======================================")

print(f"\nArquivos salvos em: {OUTPUT_DIR}")

print("\nEstrutura criada:")

print(f"""
analysis/
├── statistics.csv
├── class_distribution.png
├── histograms/
├── boxplots/
├── correlation/
│   ├── correlation_matrix.csv
│   ├── correlation_heatmap.png
│   ├── label_correlations.csv
│   └── label_correlations.png
└── label_analysis/
""")
