import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ==========================
# CONFIG
# ==========================

USER_REPO_DATASET = "./data/processed/user_repo_dataset.csv"

USER_TFIDF = "user_tfidf.csv"
REPO_TFIDF = "repos_tfidf.csv"

OUTPUT = "user_repo_dataset_with_text_similarity.csv"


# ==========================
# Carregar arquivos
# ==========================

dataset = pd.read_csv(USER_REPO_DATASET)

user_vectors = pd.read_csv(USER_TFIDF)
repo_vectors = pd.read_csv(REPO_TFIDF)


# ==========================
# Indexar vetores
# ==========================

user_vectors = user_vectors.set_index("username")
repo_vectors = repo_vectors.set_index("repo_name")


# ==========================
# Similaridade
# ==========================

similarities = []

for _, row in dataset.iterrows():
    username = row["username"]
    repo = row["repo_name"]

    try:
        user_vec = user_vectors.loc[username].values.reshape(1, -1)
        repo_vec = repo_vectors.loc[repo].values.reshape(1, -1)

        sim = cosine_similarity(user_vec, repo_vec)[0][0]

    except KeyError:
        sim = np.nan

    similarities.append(sim)


dataset["bio_repo_cosine_similarity"] = similarities


# ==========================
# Salvar cópia
# ==========================

dataset.to_csv(OUTPUT, index=False)

print(f"Arquivo salvo: {OUTPUT}")

print(dataset[["username", "repo_name", "bio_repo_cosine_similarity"]].head())
