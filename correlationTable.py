import os
import json
import random
import pandas as pd
import numpy as np

# =========================================================
# CONFIGURAÇÕES
# =========================================================

USERS_CSV = "data/raw/users.csv"
REPOS_CSV = "data/raw/repos.csv"

OUTPUT_DIR = "data/processed"
OUTPUT_FILE = f"{OUTPUT_DIR}/user_repo_dataset.csv"

# quantidade de negativos = positivos
NEGATIVE_RATIO = 1

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =========================================================
# PASTAS
# =========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================


def safe_json_load(value):
    if pd.isna(value):
        return []

    # já é lista
    if isinstance(value, list):
        return value

    # string
    if isinstance(value, str):
        value = value.strip()

        # ============================================
        # tenta JSON
        # ============================================

        try:
            parsed = json.loads(value)

            if isinstance(parsed, list):
                return parsed

        except:
            pass

        # ============================================
        # separador |
        # ============================================

        if "|" in value:
            return [v.strip() for v in value.split("|") if v.strip()]

        # ============================================
        # separador ,
        # ============================================

        if "," in value:
            return [v.strip() for v in value.split(",") if v.strip()]

        # ============================================
        # valor único
        # ============================================

        if len(value) > 0:
            return [value]

    return []


def jaccard_similarity(set_a, set_b):
    union = set_a.union(set_b)

    if len(union) == 0:
        return 0

    return len(set_a.intersection(set_b)) / len(union)


def overlap_count(set_a, set_b):
    return len(set_a.intersection(set_b))


def safe_div(a, b):
    if b == 0 or pd.isna(b):
        return 0

    return a / b


# =========================================================
# CARREGAR DADOS
# =========================================================

print("\nCarregando datasets...")

users_df = pd.read_csv(USERS_CSV)

repos_df = pd.read_csv(REPOS_CSV)

print(f"\nUsers shape: {users_df.shape}")

print(f"Repos shape: {repos_df.shape}")

# =========================================================
# CONVERSÃO NUMÉRICA
# =========================================================

print("\nConvertendo colunas numéricas...")

numeric_columns_users = [
    "followers",
    "following",
    "follower_following_ratio",
    "public_repos",
    "public_gists",
    "total_stars_received",
    "total_forks_owned",
    "total_watchers",
    "total_open_issues",
    "num_languages",
    "bio_length",
    "push_events",
    "pr_events",
    "issue_events",
    "fork_events",
    "watch_events",
    "create_events",
    "comment_events",
    "total_recent_events",
    "starred_repos_count",
    "watching_repos_count",
    "account_age_days",
]

for col in numeric_columns_users:
    if col in users_df.columns:
        users_df[col] = pd.to_numeric(users_df[col], errors="coerce")

numeric_columns_repos = [
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

for col in numeric_columns_repos:
    if col in repos_df.columns:
        repos_df[col] = pd.to_numeric(repos_df[col], errors="coerce")

# =========================================================
# DATAS
# =========================================================

print("\nConvertendo datas...")

today = pd.Timestamp.now(tz="UTC")

if "last_updated" in users_df.columns:
    users_df["last_updated"] = pd.to_datetime(
        users_df["last_updated"], errors="coerce", utc=True
    )

    users_df["days_since_user_update"] = (today - users_df["last_updated"]).dt.days

    users_df["days_since_user_update"] = users_df["days_since_user_update"].fillna(9999)

# =========================================================
# JSON / LISTAS
# =========================================================

print("\nProcessando listas JSON...")

users_df["languages_list"] = users_df["languages"].apply(safe_json_load)

repos_df["languages_list"] = repos_df["languages"].apply(safe_json_load)

repos_df["topics_list"] = repos_df["topics"].apply(safe_json_load)

# =========================================================
# POSITIVOS
# =========================================================

print("\nGerando pares positivos...")

positive_pairs = []

users_set = set(users_df["username"])

for _, repo in repos_df.iterrows():
    owner = repo["owner"]

    if owner in users_set:
        positive_pairs.append(
            {"username": owner, "repo_name": repo["repo_name"], "label": 1}
        )

print(f"Pares positivos: {len(positive_pairs)}")

# =========================================================
# SET POSITIVOS
# =========================================================

positive_set = set((p["username"], p["repo_name"]) for p in positive_pairs)

# =========================================================
# NEGATIVOS BALANCEADOS
# =========================================================

print("\nGerando pares negativos balanceados...")

all_users = users_df["username"].tolist()

all_repos = repos_df["repo_name"].tolist()

target_negatives = len(positive_pairs) * NEGATIVE_RATIO

negative_pairs = set()

# ---------------------------------------------------------
# MAPAS DE LINGUAGENS
# ---------------------------------------------------------

user_languages_map = {}

for _, row in users_df.iterrows():
    langs = row["languages_list"]

    if not isinstance(langs, list):
        langs = []

    user_languages_map[row["username"]] = set(langs)

repo_languages_map = {}

for _, row in repos_df.iterrows():
    langs = row["languages_list"]

    if not isinstance(langs, list):
        langs = []

    repo_languages_map[row["repo_name"]] = set(langs)

# ---------------------------------------------------------
# GERAÇÃO
# ---------------------------------------------------------

attempts = 0

max_attempts = target_negatives * 200

while len(negative_pairs) < target_negatives and attempts < max_attempts:
    attempts += 1

    user = random.choice(all_users)

    repo = random.choice(all_repos)

    pair = (user, repo)

    # evita positivos
    if pair in positive_set:
        continue

    # evita repetidos
    if pair in negative_pairs:
        continue

    user_langs = user_languages_map.get(user, set())

    repo_langs = repo_languages_map.get(repo, set())

    overlap = len(user_langs.intersection(repo_langs))

    # =====================================================
    # NEGATIVOS DIFÍCEIS
    # =====================================================

    # 70% chance -> overlap >= 1
    # 30% chance -> qualquer negativo

    use_hard_negative = random.random() < 0.7

    if use_hard_negative:
        if overlap >= 1:
            negative_pairs.add(pair)

    else:
        negative_pairs.add(pair)

# =========================================================
# FALLBACK
# =========================================================

# garante completar os negativos
while len(negative_pairs) < target_negatives:
    user = random.choice(all_users)

    repo = random.choice(all_repos)

    pair = (user, repo)

    if pair in positive_set:
        continue

    if pair in negative_pairs:
        continue

    negative_pairs.add(pair)

# =========================================================
# CONVERSÃO
# =========================================================

negative_pairs = [
    {"username": user, "repo_name": repo, "label": 0} for user, repo in negative_pairs
]

print(f"Pares negativos: {len(negative_pairs)}")

# =========================================================
# DATAFRAME DE PARES
# =========================================================

pairs_df = pd.DataFrame(positive_pairs + negative_pairs)

print(f"\nTotal de pares: {len(pairs_df)}")

# =========================================================
# MERGE USERS
# =========================================================

print("\nMerge usuários...")

dataset = pairs_df.merge(users_df, on="username", how="left")

# =========================================================
# MERGE REPOS
# =========================================================

print("\nMerge repositórios...")

dataset = dataset.merge(
    repos_df, on="repo_name", how="left", suffixes=("_user", "_repo")
)

print(f"\nDataset shape: {dataset.shape}")

# =========================================================
# FEATURES RELACIONAIS
# =========================================================

print("\nCriando features relacionais...")

final_rows = []

for _, row in dataset.iterrows():
    # =====================================================
    # LINGUAGENS
    # =====================================================

    user_languages = set(
        row["languages_list_user"]
        if isinstance(row["languages_list_user"], list)
        else []
    )

    repo_languages = set(
        row["languages_list_repo"]
        if isinstance(row["languages_list_repo"], list)
        else []
    )

    # =====================================================
    # FEATURES RELACIONAIS
    # =====================================================

    dominant_language_match = int(
        row.get("dominant_language_user", "") == row.get("dominant_language_repo", "")
    )

    main_language_match = int(row.get("main_language", "") in user_languages)

    language_overlap = overlap_count(user_languages, repo_languages)

    language_jaccard = jaccard_similarity(user_languages, repo_languages)

    # =====================================================
    # DIFERENÇAS
    # =====================================================

    stars_gap = abs(row.get("total_stars_received", 0) - row.get("repo_stars", 0))

    forks_gap = abs(row.get("total_forks_owned", 0) - row.get("repo_forks", 0))

    followers_gap = abs(row.get("followers", 0) - row.get("repo_watchers", 0))

    age_gap = abs(row.get("account_age_days", 0) - row.get("repo_age_days", 0))

    # =====================================================
    # RATIOS
    # =====================================================

    user_repo_popularity_ratio = safe_div(
        row.get("total_stars_received", 0), row.get("repo_stars", 0)
    )

    contributor_popularity_ratio = safe_div(
        row.get("followers", 0), row.get("contributors_count", 0)
    )

    # =====================================================
    # BOOLEANOS
    # =====================================================

    experienced_user = int(row.get("public_repos", 0) >= 10)

    popular_repo = int(row.get("repo_stars", 0) >= 100)

    active_user = int(row.get("days_since_user_update", 9999) <= 365)

    active_repo = int(row.get("days_since_update", 9999) <= 365)

    has_social_presence = int((row.get("has_blog", 0) + row.get("has_twitter", 0)) > 0)

    # =====================================================
    # LINHA FINAL
    # =====================================================

    final_rows.append(
        {
            # label
            "label": row["label"],
            # ids
            "username": row["username"],
            "repo_name": row["repo_name"],
            # relacionais
            "dominant_language_match": dominant_language_match,
            "main_language_match": main_language_match,
            "language_overlap": language_overlap,
            "language_jaccard": language_jaccard,
            "stars_gap": stars_gap,
            "forks_gap": forks_gap,
            "followers_gap": followers_gap,
            "age_gap": age_gap,
            "user_repo_popularity_ratio": user_repo_popularity_ratio,
            "contributor_popularity_ratio": contributor_popularity_ratio,
            # booleanos
            "experienced_user": experienced_user,
            "popular_repo": popular_repo,
            "active_user": active_user,
            "active_repo": active_repo,
            "has_social_presence": has_social_presence,
            # USER FEATURES
            "followers": row.get("followers", 0),
            "following": row.get("following", 0),
            "public_repos": row.get("public_repos", 0),
            "total_stars_received": row.get("total_stars_received", 0),
            "num_languages_user": row.get("num_languages_user", 0),
            "account_age_days": row.get("account_age_days", 0),
            "total_recent_events": row.get("total_recent_events", 0),
            # REPO FEATURES
            "repo_stars": row.get("repo_stars", 0),
            "repo_forks": row.get("repo_forks", 0),
            "repo_watchers": row.get("repo_watchers", 0),
            "repo_size": row.get("repo_size", 0),
            "contributors_count": row.get("contributors_count", 0),
            "num_languages_repo": row.get("num_languages_repo", 0),
            "repo_age_days": row.get("repo_age_days", 0),
            "days_since_update": row.get("days_since_update", 0),
        }
    )

# =========================================================
# DATAFRAME FINAL
# =========================================================

final_df = pd.DataFrame(final_rows)

# =========================================================
# LIMPEZA
# =========================================================

final_df = final_df.replace([np.inf, -np.inf], 0)

final_df = final_df.fillna(0)

# =========================================================
# SALVAR
# =========================================================

final_df.to_csv(OUTPUT_FILE, index=False)

# =========================================================
# RESULTADOS
# =========================================================

print("\n===================================")
print("DATASET RELACIONAL CRIADO")
print("===================================")

print(f"\nShape final: {final_df.shape}")

print("\nDistribuição das classes:")

print(final_df["label"].value_counts())

print(f"\nArquivo salvo em:")

print(OUTPUT_FILE)

print("\nColunas finais:")

for col in final_df.columns:
    print(f"- {col}")
