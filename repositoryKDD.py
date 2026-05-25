import os
import json
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================================================
# CONFIGURAÇÕES
# =========================================================

INPUT_CSV = "data/raw/repos.csv"

OUTPUT_DIR = "data/analysis"

STATS_DIR = f"{OUTPUT_DIR}/statistics"
HIST_DIR = f"{OUTPUT_DIR}/histograms"
BOX_DIR = f"{OUTPUT_DIR}/boxplots"
CORR_DIR = f"{OUTPUT_DIR}/correlation"
PCA_DIR = f"{OUTPUT_DIR}/pca"

# =========================================================
# PASTAS
# =========================================================

os.makedirs(STATS_DIR, exist_ok=True)
os.makedirs(HIST_DIR, exist_ok=True)
os.makedirs(BOX_DIR, exist_ok=True)
os.makedirs(CORR_DIR, exist_ok=True)
os.makedirs(PCA_DIR, exist_ok=True)

# =========================================================
# CARREGAR DATASET
# =========================================================

print("\nCarregando dataset...")

df = pd.read_csv(INPUT_CSV)

print(f"\nShape do dataset: {df.shape}")

# =========================================================
# COLUNAS NUMÉRICAS
# =========================================================

numeric_columns = [
    "repo_stars",
    "repo_forks",
    "repo_watchers",
    "repo_open_issues",
    "repo_size",
    "dominant_language_ratio",
    "num_languages",
    "num_topics",
    "contributors_count",
    "description_length",
    "repo_age_days",
    "days_since_update",
    "stars_per_contributor",
    "forks_per_contributor",
]

# =========================================================
# LIMPEZA
# =========================================================

df = df.replace([np.inf, -np.inf], np.nan)

# =========================================================
# ESTATÍSTICAS DESCRITIVAS
# =========================================================

print("\nGerando estatísticas descritivas...")

stats = df[numeric_columns].describe().T

stats["median"] = df[numeric_columns].median()

stats.to_csv(f"{STATS_DIR}/descriptive_statistics.csv")

print(stats)

# =========================================================
# HISTOGRAMAS
# =========================================================

print("\nGerando histogramas...")

for column in numeric_columns:
    plt.figure(figsize=(8, 5))

    values = df[column].dropna()

    plt.hist(values, bins=30)

    plt.title(f"Histograma - {column}")

    plt.xlabel(column)

    plt.ylabel("Frequência")

    plt.tight_layout()

    plt.savefig(f"{HIST_DIR}/{column}_histogram.png")

    plt.close()

# =========================================================
# BOXPLOTS
# =========================================================

print("\nGerando boxplots...")

for column in numeric_columns:
    plt.figure(figsize=(8, 5))

    values = df[column].dropna()

    plt.boxplot(values)

    plt.title(f"Boxplot - {column}")

    plt.ylabel(column)

    plt.tight_layout()

    plt.savefig(f"{BOX_DIR}/{column}_boxplot.png")

    plt.close()

# =========================================================
# CORRELAÇÃO
# =========================================================

print("\nGerando matriz de correlação...")

corr = df[numeric_columns].corr()

corr.to_csv(f"{CORR_DIR}/correlation_matrix.csv")

plt.figure(figsize=(14, 12))

sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Matriz de Correlação")

plt.tight_layout()

plt.savefig(f"{CORR_DIR}/correlation_heatmap.png")

plt.close()

# =========================================================
# PCA
# =========================================================

print("\nExecutando PCA...")

X = df[numeric_columns].fillna(0)

# normalização
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

# dataframe PCA
pca_df = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1]})

pca_df.to_csv(f"{PCA_DIR}/pca_projection.csv", index=False)

# =========================================================
# GRÁFICO PCA
# =========================================================

plt.figure(figsize=(10, 8))

plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.title("PCA Projection of Repository Dataset")

plt.tight_layout()

plt.savefig(f"{PCA_DIR}/pca_projection.png")

plt.close()

# =========================================================
# VARIÂNCIA EXPLICADA
# =========================================================

explained_variance = pd.DataFrame(
    {
        "Component": ["PC1", "PC2"],
        "Explained Variance Ratio": pca.explained_variance_ratio_,
    }
)

explained_variance.to_csv(f"{PCA_DIR}/explained_variance.csv", index=False)

# =========================================================
# IMPORTÂNCIA DAS FEATURES NO PCA
# =========================================================

loadings = pd.DataFrame(
    pca.components_.T, columns=["PC1", "PC2"], index=numeric_columns
)

loadings.to_csv(f"{PCA_DIR}/pca_feature_loadings.csv")

# =========================================================
# TOP CORRELAÇÕES
# =========================================================

print("\nCalculando top correlações...")

corr_pairs = corr.unstack()

corr_pairs = corr_pairs.sort_values(ascending=False)

# remove autocorrelação
corr_pairs = corr_pairs[corr_pairs != 1.0]

top_corr = corr_pairs.drop_duplicates()

top_corr.to_csv(f"{CORR_DIR}/top_correlations.csv")

# =========================================================
# RESUMO FINAL
# =========================================================

summary_path = f"{OUTPUT_DIR}/summary.txt"

with open(summary_path, "w") as f:
    f.write("=====================================\n")
    f.write("ANÁLISE EXPLORATÓRIA DO DATASET\n")
    f.write("=====================================\n\n")

    f.write(f"Shape do dataset: {df.shape}\n\n")

    f.write("Colunas numéricas analisadas:\n\n")

    for col in numeric_columns:
        f.write(f"- {col}\n")

    f.write("\n")

    f.write("Variância explicada PCA:\n\n")

    for i, var in enumerate(pca.explained_variance_ratio_):
        f.write(f"PC{i + 1}: {var:.4f}\n")

print("\n=====================================")
print("ANÁLISE FINALIZADA")
print("=====================================")

print(f"\nArquivos salvos em: {OUTPUT_DIR}")

print("\nEstrutura gerada:")

print("""
data/analysis/
├── statistics/
│   └── descriptive_statistics.csv
│
├── histograms/
│   └── *.png
│
├── boxplots/
│   └── *.png
│
├── correlation/
│   ├── correlation_matrix.csv
│   ├── correlation_heatmap.png
│   └── top_correlations.csv
│
├── pca/
│   ├── pca_projection.csv
│   ├── pca_projection.png
│   ├── explained_variance.csv
│   └── pca_feature_loadings.csv
│
└── summary.txt
""")
