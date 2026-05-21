import os
import time
import random
import requests
import pandas as pd
from dotenv import load_dotenv

# =========================================================
# CONFIGURAÇÕES
# =========================================================

load_dotenv()

TOKEN = os.getenv("GITHUB_TOKEN")

headers = {"Authorization": f"token {TOKEN}"}

# usuários iniciais
SEED_USERS = ["torvalds", "karpathy", "tensorflow", "pytorch", "huggingface"]

# quantidade
FOLLOWERS_PER_SEED = 50
REPOS_PER_USER = 10

# =========================================================
# PASTAS
# =========================================================

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================


def github_get(url):
    """
    Faz requisição para API GitHub.
    """

    response = requests.get(url, headers=headers)

    # limite API
    if response.status_code == 403:
        print("Limite da API atingido.")
        exit()

    if response.status_code != 200:
        return None

    time.sleep(0.2)

    return response.json()


def get_followers(username, limit=50):
    url = f"https://api.github.com/users/{username}/followers"

    data = github_get(url)

    if data is None:
        return []

    return [u["login"] for u in data[:limit]]


def get_user(username):
    url = f"https://api.github.com/users/{username}"

    return github_get(url)


def get_user_repos(username):
    url = f"https://api.github.com/users/{username}/repos"

    data = github_get(url)

    if data is None:
        return []

    return data


def get_repo(repo_full_name):
    url = f"https://api.github.com/repos/{repo_full_name}"

    return github_get(url)


def get_contributors(repo_full_name):
    url = f"https://api.github.com/repos/{repo_full_name}/contributors"

    data = github_get(url)

    if data is None:
        return []

    return [c["login"] for c in data]


# =========================================================
# COLETA DE USUÁRIOS
# =========================================================

print("\nColetando usuários...")

all_users = set()

for seed in SEED_USERS:
    print(f"Seed: {seed}")

    followers = get_followers(seed, FOLLOWERS_PER_SEED)

    for f in followers:
        all_users.add(f)

all_users = list(all_users)

print(f"\nTotal de usuários coletados: {len(all_users)}")

# =========================================================
# FEATURES DOS USUÁRIOS
# =========================================================

users_data = []

for username in all_users:
    print(f"Usuário: {username}")

    user = get_user(username)

    if user is None:
        continue

    repos = get_user_repos(username)

    total_stars = 0
    languages = set()

    for repo in repos:
        total_stars += repo.get("stargazers_count", 0)

        lang = repo.get("language")

        if lang is not None:
            languages.add(lang)

    users_data.append(
        {
            "username": username,
            "followers": user.get("followers", 0),
            "following": user.get("following", 0),
            "public_repos": user.get("public_repos", 0),
            "public_gists": user.get("public_gists", 0),
            "account_type": user.get("type", "User"),
            "bio_length": 0 if user.get("bio") is None else len(user.get("bio")),
            "has_company": 0 if user.get("company") is None else 1,
            "total_stars": total_stars,
            "num_languages": len(languages),
        }
    )

users_df = pd.DataFrame(users_data)

users_df.to_csv("data/raw/users.csv", index=False)

print("\nArquivo salvo:")
print("data/raw/users.csv")

# =========================================================
# COLETA DE REPOSITÓRIOS
# =========================================================

print("\nColetando repositórios...")

repos_set = set()

for username in all_users:
    repos = get_user_repos(username)

    for repo in repos[:REPOS_PER_USER]:
        repos_set.add(repo["full_name"])

repos_set = list(repos_set)

print(f"\nTotal de repositórios: {len(repos_set)}")

# =========================================================
# FEATURES DOS REPOSITÓRIOS
# =========================================================

repos_data = []

repo_contributors = {}

for repo_name in repos_set:
    print(f"Repo: {repo_name}")

    repo = get_repo(repo_name)

    if repo is None:
        continue

    contributors = get_contributors(repo_name)

    repo_contributors[repo_name] = contributors

    repos_data.append(
        {
            "repo_name": repo_name,
            "repo_stars": repo.get("stargazers_count", 0),
            "repo_forks": repo.get("forks_count", 0),
            "repo_watchers": repo.get("watchers_count", 0),
            "repo_open_issues": repo.get("open_issues_count", 0),
            "repo_size": repo.get("size", 0),
            "repo_language": repo.get("language", "Unknown"),
            "contributors_count": len(contributors),
        }
    )

repos_df = pd.DataFrame(repos_data)

repos_df.to_csv("data/raw/repos.csv", index=False)

print("\nArquivo salvo:")
print("data/raw/repos.csv")

# =========================================================
# CONSTRUÇÃO DO DATASET FINAL
# =========================================================

print("\nConstruindo dataset final...")

dataset = []

for _, user in users_df.iterrows():
    username = user["username"]

    for _, repo in repos_df.iterrows():
        repo_name = repo["repo_name"]

        contributors = repo_contributors.get(repo_name, [])

        # LABEL
        label = 1 if username in contributors else 0

        # FEATURE RELACIONAL
        language_match = 0

        user_repos = get_user_repos(username)

        user_languages = set()

        for r in user_repos:
            lang = r.get("language")

            if lang is not None:
                user_languages.add(lang)

        if repo["repo_language"] in user_languages:
            language_match = 1

        dataset.append(
            {
                # usuário
                "followers": user["followers"],
                "following": user["following"],
                "public_repos": user["public_repos"],
                "public_gists": user["public_gists"],
                "bio_length": user["bio_length"],
                "has_company": user["has_company"],
                "total_stars": user["total_stars"],
                "num_languages": user["num_languages"],
                # repo
                "repo_stars": repo["repo_stars"],
                "repo_forks": repo["repo_forks"],
                "repo_watchers": repo["repo_watchers"],
                "repo_open_issues": repo["repo_open_issues"],
                "repo_size": repo["repo_size"],
                "contributors_count": repo["contributors_count"],
                # relacional
                "language_match": language_match,
                # target
                "label": label,
            }
        )

dataset_df = pd.DataFrame(dataset)

# =========================================================
# BALANCEAMENTO SIMPLES
# =========================================================

positive = dataset_df[dataset_df["label"] == 1]
negative = dataset_df[dataset_df["label"] == 0]

negative = negative.sample(n=min(len(negative), len(positive) * 2), random_state=42)

dataset_df = pd.concat([positive, negative])

dataset_df = dataset_df.sample(frac=1, random_state=42)

# =========================================================
# SALVAR DATASET
# =========================================================

dataset_df.to_csv("data/processed/dataset.csv", index=False)

print("\nDataset final salvo:")
print("data/processed/dataset.csv")

print("\nResumo:")

print(dataset_df["label"].value_counts())

print("\nShape:")
print(dataset_df.shape)
