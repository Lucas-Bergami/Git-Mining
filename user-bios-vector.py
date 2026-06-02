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

INPUT_CSV  = "data/raw/users_bios.csv"
OUTPUT_DIR = "data/analysis/tfidf"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parâmetros do TF-IDF (ajustados automaticamente ao tamanho do dataset)
MAX_FEATURES  = 500   # vocabulário máximo
MAX_DF        = 0.95  # ignora termos que aparecem em mais de 95% dos docs
NGRAM_RANGE   = (1, 2) # unigramas e bigramas

# Parâmetros do clustering
N_CLUSTERS = 5

# =========================================================
# LEITURA E FILTRAGEM
# =========================================================

print("\nLendo bios...")
df = pd.read_csv(INPUT_CSV)

print(f"Total de usuários   : {len(df)}")

# Remove bios vazias
df_valid = df[df["bio"].notna() & (df["bio"].str.strip() != "")].copy()
df_valid = df_valid.reset_index(drop=True)

print(f"Com bio             : {len(df_valid)}")
print(f"Sem bio (ignorados) : {len(df) - len(df_valid)}")

if len(df_valid) == 0:
    print("\nNenhuma bio disponível para análise.")
    exit()

# =========================================================
# TF-IDF
# =========================================================

print("\nAplicando TF-IDF...")

# min_df adaptativo: 1 se poucos docs, 2 se suficiente
min_df = 1 if len(df_valid) < 20 else 2

vectorizer = TfidfVectorizer(
    max_features = MAX_FEATURES,
    min_df       = min_df,
    max_df       = MAX_DF,
    ngram_range  = NGRAM_RANGE,
    sublinear_tf = True,   # aplica log(tf) — reduz peso de termos muito frequentes
)

user_tfidf = vectorizer.fit_transform(df_valid["bio"])
feature_names = vectorizer.get_feature_names_out()

print(f"Shape da matriz TF-IDF : {user_tfidf.shape}")
print(f"  → {user_tfidf.shape[0]} usuários × {user_tfidf.shape[1]} termos")

# =========================================================
# TOP TERMOS GLOBAIS
# =========================================================

print("\nCalculando top termos globais...")

# Soma os scores TF-IDF de cada termo em todos os documentos
term_scores = user_tfidf.sum(axis=0).A1
top_idx     = term_scores.argsort()[::-1][:30]

top_terms_df = pd.DataFrame({
    "termo":     feature_names[top_idx],
    "tfidf_sum": term_scores[top_idx].round(4),
})
top_terms_df.to_csv(f"{OUTPUT_DIR}/top_termos.csv", index=False)

print(top_terms_df.head(15).to_string(index=False))

# Gráfico top termos
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top_terms_df["termo"][:20][::-1], top_terms_df["tfidf_sum"][:20][::-1],
        color="steelblue")
ax.set_title("Top 20 Termos por Score TF-IDF (bios dos usuários)", fontweight="bold")
ax.set_xlabel("Soma TF-IDF")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/top_termos.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[salvo] {OUTPUT_DIR}/top_termos.png")

# =========================================================
# MATRIZ TF-IDF → CSV (usuário × top termos)
# =========================================================

print("\nSalvando matriz TF-IDF (top termos)...")

# Usa apenas os top 50 termos para não explodir o CSV
top50_idx  = term_scores.argsort()[::-1][:50]
top50_names = feature_names[top50_idx]

tfidf_dense = pd.DataFrame(
    user_tfidf[:, top50_idx].toarray(),
    columns=top50_names,
)
tfidf_dense.insert(0, "username", df_valid["username"].values)
tfidf_dense.to_csv(f"{OUTPUT_DIR}/user_tfidf.csv", index=False)
print(f"[salvo] {OUTPUT_DIR}/user_tfidf.csv")

# =========================================================
# REDUÇÃO DE DIMENSIONALIDADE (SVD / LSA)
# =========================================================

print("\nReduzindo dimensionalidade com SVD (LSA)...")

n_components = min(50, user_tfidf.shape[1] - 1, user_tfidf.shape[0] - 1)
N_CLUSTERS   = min(N_CLUSTERS, len(df_valid) - 1)  # não pode ter mais clusters que docs
svd = TruncatedSVD(n_components=n_components, random_state=42)
X_reduced = svd.fit_transform(user_tfidf)
X_reduced = normalize(X_reduced)

explained = svd.explained_variance_ratio_.cumsum()
print(f"Variância explicada (50 componentes): {explained[-1]*100:.1f}%")

# Gráfico variância explicada
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(explained) + 1), explained * 100, marker="o", markersize=3)
ax.axhline(80, color="red", linestyle="--", linewidth=1, label="80%")
ax.set_title("Variância Explicada Acumulada — SVD", fontweight="bold")
ax.set_xlabel("Número de Componentes")
ax.set_ylabel("Variância Explicada (%)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/svd_variancia.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[salvo] {OUTPUT_DIR}/svd_variancia.png")

# =========================================================
# PROJEÇÃO 2D (PC1 × PC2)
# =========================================================

print("\nGerando projeção 2D...")

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.6, s=30, color="steelblue")

# Anota os 10 usuários mais "extremos" (maior norma na projeção)
norms   = np.linalg.norm(X_reduced[:, :2], axis=1)
top10   = norms.argsort()[::-1][:10]
for idx in top10:
    ax.annotate(
        df_valid.iloc[idx]["username"],
        (X_reduced[idx, 0], X_reduced[idx, 1]),
        fontsize=7, alpha=0.8,
    )

ax.set_title("Projeção 2D das Bios (SVD — PC1 × PC2)", fontweight="bold")
ax.set_xlabel("Componente 1")
ax.set_ylabel("Componente 2")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/projecao_2d.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[salvo] {OUTPUT_DIR}/projecao_2d.png")

# =========================================================
# CLUSTERING K-MEANS
# =========================================================

print(f"\nAgrupando usuários em {N_CLUSTERS} clusters (K-Means)...")

if N_CLUSTERS < 2:
    print("Poucos usuários com bio para clustering. Pulando K-Means.")
    exit()
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
clusters = km.fit_predict(X_reduced)
df_valid["cluster"] = clusters

# Top termos por cluster (a partir dos centroides no espaço SVD → projeta de volta)
print("\nTop termos por cluster:")
for c in range(N_CLUSTERS):
    cluster_docs = user_tfidf[clusters == c]
    scores       = cluster_docs.sum(axis=0).A1
    top_c        = feature_names[scores.argsort()[::-1][:8]]
    n_users      = (clusters == c).sum()
    print(f"  Cluster {c} ({n_users} usuários): {', '.join(top_c)}")

# Salva usuários com cluster
cluster_out = df_valid[["username", "bio", "cluster"]].copy()
cluster_out.to_csv(f"{OUTPUT_DIR}/usuarios_clusters.csv", index=False)
print(f"\n[salvo] {OUTPUT_DIR}/usuarios_clusters.csv")

# Gráfico projeção 2D colorida por cluster
colors = plt.cm.tab10.colors
fig, ax = plt.subplots(figsize=(10, 7))
for c in range(N_CLUSTERS):
    mask = clusters == c
    ax.scatter(
        X_reduced[mask, 0], X_reduced[mask, 1],
        label=f"Cluster {c} (n={mask.sum()})",
        alpha=0.7, s=40, color=colors[c],
    )
ax.set_title("Clusters de Usuários por Bio (K-Means + SVD)", fontweight="bold")
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
print(f"  top_termos.csv       → ranking global de termos")
print(f"  top_termos.png       → gráfico top 20 termos")
print(f"  user_tfidf.csv       → matriz usuário × top 50 termos")
print(f"  svd_variancia.png    → variância explicada acumulada")
print(f"  projecao_2d.png      → projeção 2D das bios")
print(f"  usuarios_clusters.csv→ usuários com cluster atribuído")
print(f"  clusters.png         → projeção 2D colorida por cluster")