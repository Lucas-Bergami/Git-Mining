import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# =========================================================
# CONFIGURAÇÕES
# =========================================================

INPUT_CSV  = "data/raw/users_texts.csv"
OUTPUT_DIR = "data/analysis/tfidf"

os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_FEATURES = 500
MAX_DF       = 0.95
NGRAM_RANGE  = (1, 2)
N_CLUSTERS   = 5

# =========================================================
# LEITURA E FILTRAGEM
# =========================================================

print("\nLendo textos (bio + README)...")
df = pd.read_csv(INPUT_CSV)

print(f"Total de usuários     : {len(df)}")
print(f"Com bio               : {df['bio'].notna().sum()}")
print(f"Com README            : {df['readme'].notna().sum()}")

# Usa a coluna "text" (bio + readme combinados) — remove quem não tem nenhum
df_valid = df[df["text"].notna() & (df["text"].str.strip() != "")].copy()
df_valid = df_valid.reset_index(drop=True)

print(f"Com algum texto       : {len(df_valid)}")
print(f"Sem nenhum texto      : {len(df) - len(df_valid)} (ignorados)")

if len(df_valid) == 0:
    print("\nNenhum texto disponível para análise.")
    exit()

# =========================================================
# TF-IDF
# =========================================================

print("\nAplicando TF-IDF (bio + README combinados)...")

min_df = 1 if len(df_valid) < 20 else 2

vectorizer = TfidfVectorizer(
    max_features = MAX_FEATURES,
    min_df       = min_df,
    max_df       = MAX_DF,
    ngram_range  = NGRAM_RANGE,
    sublinear_tf = True,
)

tfidf_matrix  = vectorizer.fit_transform(df_valid["text"])
feature_names = vectorizer.get_feature_names_out()

print(f"Shape da matriz TF-IDF : {tfidf_matrix.shape}")
print(f"  → {tfidf_matrix.shape[0]} usuários × {tfidf_matrix.shape[1]} termos")

# =========================================================
# TOP TERMOS GLOBAIS
# =========================================================

print("\nCalculando top termos globais...")

term_scores = tfidf_matrix.sum(axis=0).A1
top_idx     = term_scores.argsort()[::-1][:30]

top_terms_df = pd.DataFrame({
    "termo":     feature_names[top_idx],
    "tfidf_sum": term_scores[top_idx].round(4),
})
top_terms_df.to_csv(f"{OUTPUT_DIR}/top_termos.csv", index=False)

print(top_terms_df.head(15).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top_terms_df["termo"][:20][::-1], top_terms_df["tfidf_sum"][:20][::-1],
        color="steelblue")
ax.set_title("Top 20 Termos por Score TF-IDF (Bio + README)", fontweight="bold")
ax.set_xlabel("Soma TF-IDF")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/top_termos.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[salvo] {OUTPUT_DIR}/top_termos.png")

# =========================================================
# MATRIZ TF-IDF → CSV (usuário × top 50 termos)
# =========================================================

print("\nSalvando matriz TF-IDF (top 50 termos)...")

top50_idx   = term_scores.argsort()[::-1][:50]
top50_names = feature_names[top50_idx]

tfidf_dense = pd.DataFrame(
    tfidf_matrix[:, top50_idx].toarray(),
    columns=top50_names,
)
tfidf_dense.insert(0, "username", df_valid["username"].values)
tfidf_dense.to_csv(f"{OUTPUT_DIR}/tfidf_matrix.csv", index=False)
print(f"[salvo] {OUTPUT_DIR}/tfidf_matrix.csv")

# =========================================================
# REDUÇÃO DE DIMENSIONALIDADE (SVD / LSA)
# =========================================================

print("\nReduzindo dimensionalidade com SVD (LSA)...")

n_components = min(50, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
N_CLUSTERS   = min(N_CLUSTERS, len(df_valid) - 1)

svd       = TruncatedSVD(n_components=n_components, random_state=42)
X_reduced = svd.fit_transform(tfidf_matrix)
X_reduced = normalize(X_reduced)

explained = svd.explained_variance_ratio_.cumsum()
print(f"Variância explicada ({n_components} componentes): {explained[-1]*100:.1f}%")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(explained) + 1), explained * 100,
        marker="o", markersize=3, color="steelblue")
ax.axhline(80, color="red", linestyle="--", linewidth=1, label="80%")
ax.set_title("Variância Explicada Acumulada — SVD (Bio + README)", fontweight="bold")
ax.set_xlabel("Número de Componentes")
ax.set_ylabel("Variância Explicada (%)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/svd_variancia.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[salvo] {OUTPUT_DIR}/svd_variancia.png")

# =========================================================
# PROJEÇÃO 2D
# =========================================================

print("\nGerando projeção 2D...")

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.6, s=40, color="steelblue")

norms = np.linalg.norm(X_reduced[:, :2], axis=1)
top10 = norms.argsort()[::-1][:10]
for idx in top10:
    ax.annotate(
        df_valid.iloc[idx]["username"],
        (X_reduced[idx, 0], X_reduced[idx, 1]),
        fontsize=7, alpha=0.8,
    )

ax.set_title("Projeção 2D dos Textos de Perfil (SVD — PC1 × PC2)", fontweight="bold")
ax.set_xlabel("Componente 1")
ax.set_ylabel("Componente 2")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/projecao_2d.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[salvo] {OUTPUT_DIR}/projecao_2d.png")

# =========================================================
# CLUSTERING K-MEANS
# =========================================================

if N_CLUSTERS < 2:
    print("\nPoucos usuários com texto para clustering. Pulando K-Means.")
    exit()

print(f"\nAgrupando usuários em {N_CLUSTERS} clusters (K-Means)...")

km       = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
clusters = km.fit_predict(X_reduced)
df_valid["cluster"] = clusters

print("\nTop termos por cluster:")
for c in range(N_CLUSTERS):
    cluster_docs = tfidf_matrix[clusters == c]
    scores       = cluster_docs.sum(axis=0).A1
    top_c        = feature_names[scores.argsort()[::-1][:8]]
    n_users      = (clusters == c).sum()
    print(f"  Cluster {c} ({n_users} usuários): {', '.join(top_c)}")

cluster_out = df_valid[["username", "bio", "readme", "cluster"]].copy()
cluster_out.to_csv(f"{OUTPUT_DIR}/usuarios_clusters.csv", index=False)
print(f"\n[salvo] {OUTPUT_DIR}/usuarios_clusters.csv")

colors = plt.cm.tab10.colors
fig, ax = plt.subplots(figsize=(10, 7))
for c in range(N_CLUSTERS):
    mask = clusters == c
    ax.scatter(
        X_reduced[mask, 0], X_reduced[mask, 1],
        label=f"Cluster {c} (n={mask.sum()})",
        alpha=0.7, s=50, color=colors[c],
    )
    for idx in np.where(mask)[0]:
        ax.annotate(
            df_valid.iloc[idx]["username"],
            (X_reduced[idx, 0], X_reduced[idx, 1]),
            fontsize=6, alpha=0.6,
        )

ax.set_title("Clusters de Usuários por Texto de Perfil (K-Means + SVD)", fontweight="bold")
ax.set_xlabel("Componente 1")
ax.set_ylabel("Componente 2")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/clusters.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[salvo] {OUTPUT_DIR}/clusters.png")

# =========================================================
# RESUMO
# =========================================================

print("\n" + "=" * 60)
print("RESUMO FINAL")
print("=" * 60)
print(f"Usuários analisados    : {len(df_valid)}")
print(f"Vocabulário TF-IDF     : {len(feature_names)} termos")
print(f"Componentes SVD        : {n_components}")
print(f"Clusters K-Means       : {N_CLUSTERS}")
print(f"\nArquivos em {OUTPUT_DIR}/")
print(f"  top_termos.csv        → ranking global de termos")
print(f"  top_termos.png        → gráfico top 20 termos")
print(f"  tfidf_matrix.csv      → matriz usuário × top 50 termos")
print(f"  svd_variancia.png     → variância explicada acumulada")
print(f"  projecao_2d.png       → projeção 2D dos textos")
print(f"  usuarios_clusters.csv → usuários com bio, readme e cluster")
print(f"  clusters.png          → projeção 2D colorida por cluster")