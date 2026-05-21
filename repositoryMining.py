import os
import json
import time
import requests
import pandas as pd

from dotenv import load_dotenv
from datetime import datetime, timezone

# =========================================================
# CONFIGURAÇÕES
# =========================================================

load_dotenv()

TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github+json"}

INPUT_CSV = "data/raw/users.csv"
OUTPUT_CSV = "data/raw/repos.csv"
PROGRESS_FILE = "data/raw/progress.json"

# quantidade máxima de repos por usuário
MAX_REPOS_PER_USER = 1

# filtros de qualidade
MIN_STARS = 5
MIN_SIZE = 50

# =========================================================
# PASTAS
# =========================================================

os.makedirs("data/raw", exist_ok=True)

# =========================================================
# PROGRESSO
# =========================================================


def save_progress(username):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"last_username": username}, f)


def load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return None

    with open(PROGRESS_FILE, "r") as f:
        data = json.load(f)

    return data.get("last_username")


# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================


def save_partial_csv(repos_data):
    temp_df = pd.DataFrame(repos_data)

    temp_df.to_csv(OUTPUT_CSV, index=False)

    print("\n[CHECKPOINT] CSV salvo.")


def github_get(url, repos_data=None):
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)

    except Exception as e:
        print(f"\nErro de conexão: {e}")

        if repos_data is not None:
            save_partial_csv(repos_data)

        exit()

    # =====================================================
    # RATE LIMIT
    # =====================================================

    remaining = int(response.headers.get("X-RateLimit-Remaining", 0))

    print(f"Requests restantes: {remaining}")

    if remaining <= 1:
        print("\n[LIMIT] Limite da API atingido.")

        reset_timestamp = int(response.headers.get("X-RateLimit-Reset", 0))

        reset_time = datetime.fromtimestamp(reset_timestamp)

        print(f"Reset em: {reset_time}")

        if repos_data is not None:
            save_partial_csv(repos_data)

        exit()

    # =====================================================
    # STATUS
    # =====================================================

    if response.status_code != 200:
        print(f"\nErro {response.status_code} em:")

        print(url)

        return None

    time.sleep(0.15)

    return response.json()


def calculate_days(date_str):
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")

        date_obj = date_obj.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)

        return (now - date_obj).days

    except:
        return 0


# =========================================================
# API GITHUB
# =========================================================


def get_user_repos(username, repos_data):
    repos = []

    page = 1

    while True:
        url = f"https://api.github.com/users/{username}/repos?per_page=100&page={page}"

        data = github_get(url, repos_data)

        if data is None or len(data) == 0:
            break

        repos.extend(data)

        page += 1

    return repos


def get_repo_languages(full_name, repos_data):
    url = f"https://api.github.com/repos/{full_name}/languages"

    data = github_get(url, repos_data)

    if data is None:
        return {}

    return data


def get_repo_topics(full_name, repos_data):
    url = f"https://api.github.com/repos/{full_name}/topics"

    response = requests.get(
        url, headers={**HEADERS, "Accept": "application/vnd.github.mercy-preview+json"}
    )

    remaining = int(response.headers.get("X-RateLimit-Remaining", 0))

    print(f"Requests restantes: {remaining}")

    if remaining <= 1:
        print("\n[LIMIT] Limite da API atingido.")

        save_partial_csv(repos_data)

        exit()

    if response.status_code != 200:
        return []

    time.sleep(0.15)

    data = response.json()

    return data.get("names", [])


def get_contributors_count(full_name, repos_data):
    url = f"https://api.github.com/repos/{full_name}/contributors?per_page=100"

    data = github_get(url, repos_data)

    if data is None:
        return 0

    return len(data)


# =========================================================
# CARREGAR USUÁRIOS
# =========================================================

users_df = pd.read_csv(INPUT_CSV)

usernames = users_df["username"].dropna().unique()

print(f"\nUsuários carregados: {len(usernames)}")

# =========================================================
# RECUPERAR CSV EXISTENTE
# =========================================================

repos_data = []

existing_repos = set()

if os.path.exists(OUTPUT_CSV):
    existing_df = pd.read_csv(OUTPUT_CSV)

    repos_data = existing_df.to_dict(orient="records")

    existing_repos = set(existing_df["repo_name"])

    print(f"\nRepos já existentes: {len(existing_repos)}")

# =========================================================
# RECUPERAR PROGRESSO
# =========================================================

last_username = load_progress()

if last_username is not None:
    print(f"\nContinuando após: {last_username}")

# =========================================================
# COLETA
# =========================================================

start_processing = last_username is None

for username in usernames:
    # =====================================================
    # RETOMADA
    # =====================================================

    if not start_processing:
        if username == last_username:
            start_processing = True

        continue

    print("\n=================================")
    print(f"Usuário: {username}")
    print("=================================")

    try:
        repos = get_user_repos(username, repos_data)

    except Exception as e:
        print(f"\nErro coletando repos: {e}")

        save_partial_csv(repos_data)

        exit()

    # =====================================================
    # FILTRAGEM
    # =====================================================

    valid_repos = []

    for repo in repos:
        if repo.get("fork", False):
            continue

        if repo.get("size", 0) < MIN_SIZE:
            continue

        if repo.get("stargazers_count", 0) < MIN_STARS:
            continue

        if repo.get("language") is None:
            continue

        valid_repos.append(repo)

    # =====================================================
    # ORDENA POR RELEVÂNCIA
    # =====================================================

    valid_repos = sorted(
        valid_repos,
        key=lambda r: (r.get("stargazers_count", 0), r.get("forks_count", 0)),
        reverse=True,
    )

    valid_repos = valid_repos[:MAX_REPOS_PER_USER]

    print(f"Repos válidos: {len(valid_repos)}")

    # =====================================================
    # FEATURES DOS REPOS
    # =====================================================

    for repo in valid_repos:
        full_name = repo["full_name"]

        if full_name in existing_repos:
            continue

        existing_repos.add(full_name)

        print(f"\nRepo: {full_name}")

        # =================================================
        # LINGUAGENS
        # =================================================

        languages_data = get_repo_languages(full_name, repos_data)

        languages = list(languages_data.keys())

        total_language_bytes = sum(languages_data.values())

        dominant_language = "Unknown"

        if len(languages_data) > 0:
            dominant_language = max(languages_data, key=languages_data.get)

        dominant_language_ratio = 0

        if total_language_bytes > 0:
            dominant_language_ratio = (
                languages_data[dominant_language] / total_language_bytes
            )

        # =================================================
        # TOPICS
        # =================================================

        topics = get_repo_topics(full_name, repos_data)

        # =================================================
        # CONTRIBUTORS
        # =================================================

        contributors_count = get_contributors_count(full_name, repos_data)

        # =================================================
        # DATAS
        # =================================================

        repo_age_days = calculate_days(repo.get("created_at"))

        days_since_update = calculate_days(repo.get("updated_at"))

        # =================================================
        # DATASET
        # =================================================

        repos_data.append(
            {
                # identificação
                "repo_name": full_name,
                "repo_id": repo.get("id", 0),
                "owner": repo["owner"]["login"],
                # popularidade
                "repo_stars": repo.get("stargazers_count", 0),
                "repo_forks": repo.get("forks_count", 0),
                "repo_watchers": repo.get("watchers_count", 0),
                # atividade
                "repo_open_issues": repo.get("open_issues_count", 0),
                # tamanho
                "repo_size": repo.get("size", 0),
                # linguagem
                "main_language": repo.get("language", "Unknown"),
                "dominant_language": dominant_language,
                "dominant_language_ratio": dominant_language_ratio,
                "languages": json.dumps(languages),
                "num_languages": len(languages),
                # tópicos
                "topics": json.dumps(topics),
                "num_topics": len(topics),
                # colaboração
                "contributors_count": contributors_count,
                # descrição
                "has_description": 0 if repo.get("description") is None else 1,
                "description_length": 0
                if repo.get("description") is None
                else len(repo.get("description")),
                # temporal
                "repo_age_days": repo_age_days,
                "days_since_update": days_since_update,
                # flags
                "has_wiki": int(repo.get("has_wiki", False)),
                "has_pages": int(repo.get("has_pages", False)),
                "has_projects": int(repo.get("has_projects", False)),
                "has_downloads": int(repo.get("has_downloads", False)),
                "has_issues": int(repo.get("has_issues", False)),
                # métricas derivadas
                "stars_per_contributor": repo.get("stargazers_count", 0)
                / max(contributors_count, 1),
                "forks_per_contributor": repo.get("forks_count", 0)
                / max(contributors_count, 1),
                # urls
                "html_url": repo.get("html_url", ""),
                "api_url": repo.get("url", ""),
            }
        )

    # =====================================================
    # CHECKPOINT
    # =====================================================

    save_partial_csv(repos_data)

    save_progress(username)

    print(f"\nCheckpoint salvo após: {username}")

# =========================================================
# FINAL
# =========================================================

final_df = pd.DataFrame(repos_data)

final_df = final_df.drop_duplicates(subset=["repo_name"])

final_df.to_csv(OUTPUT_CSV, index=False)

print("\n=================================")
print("COLETA FINALIZADA")
print("=================================")

print(f"\nRepos coletados: {len(final_df)}")

print("\nShape:")
print(final_df.shape)

print(f"\nArquivo salvo em:")
print(OUTPUT_CSV)
