import os
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

headers = {"Authorization": f"token {TOKEN}"}

SEED_USERS = ["torvalds", "karpathy", "tensorflow", "pytorch", "huggingface"]

FOLLOWERS_PER_SEED = 25
MAX_REPOS = 2  # busca apenas os top 10 repos por usuário

# =========================================================
# PASTAS
# =========================================================

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/checkpoints", exist_ok=True)

# =========================================================
# CONTROLE DE RATE LIMIT
# =========================================================

def check_rate_limit():
    """Consulta o rate limit atual da API GitHub."""
    response = requests.get("https://api.github.com/rate_limit", headers=headers)
    if response.status_code == 200:
        data = response.json()
        remaining = data["rate"]["remaining"]
        reset_at = data["rate"]["reset"]
        return remaining, reset_at
    return None, None


def save_checkpoint(data, filename):
    """Salva um checkpoint em CSV para não perder progresso."""
    path = f"data/checkpoints/{filename}.csv"
    pd.DataFrame(data if isinstance(data, list) else [data]).to_csv(path, index=False)
    print(f"  [checkpoint] Salvo: {path}")


def load_checkpoint(filename):
    """Carrega um checkpoint salvo anteriormente."""
    path = f"data/checkpoints/{filename}.csv"
    if os.path.exists(path):
        data = pd.read_csv(path).to_dict(orient="records")
        print(f"  [checkpoint] Carregado: {path} ({len(data)} registros)")
        return data
    return None


def github_get(url, params=None):
    """
    Faz requisição para API GitHub com tratamento de rate limit.
    Lança RateLimitException quando o limite é atingido.
    """
    response = requests.get(url, headers=headers, params=params)

    if response.status_code in (403, 429):
        remaining, reset_at = check_rate_limit()
        reset_time = datetime.fromtimestamp(reset_at).strftime("%H:%M:%S") if reset_at else "?"
        print(f"\n[RATE LIMIT] Restantes: {remaining}. Reset às {reset_time}.")
        print("[RATE LIMIT] Salvando checkpoints e encerrando com segurança...\n")
        raise RateLimitException("Rate limit da API GitHub atingido.")

    if response.status_code == 204:
        return []

    if response.status_code != 200:
        return None

    time.sleep(0.3)
    return response.json()


class RateLimitException(Exception):
    pass


# =========================================================
# FUNÇÕES DE COLETA
# =========================================================

def get_followers(username, limit=50):
    url = f"https://api.github.com/users/{username}/followers"
    data = github_get(url, params={"per_page": limit})
    return [u["login"] for u in (data or [])[:limit]]


def get_user(username):
    return github_get(f"https://api.github.com/users/{username}")


def get_user_repos(username):
    data = github_get(
        f"https://api.github.com/users/{username}/repos",
        params={"per_page": MAX_REPOS, "sort": "stars"},
    )
    return data or []


def get_user_events(username):
    """Eventos públicos recentes (commits, PRs, issues, etc.)."""
    data = github_get(
        f"https://api.github.com/users/{username}/events/public",
        params={"per_page": 100},
    )
    return data or []


def get_user_starred(username):
    """Repos que o usuário deu star."""
    data = github_get(
        f"https://api.github.com/users/{username}/starred",
        params={"per_page": 100},
    )
    return data or []


def get_user_subscriptions(username):
    """Repos que o usuário está watching."""
    data = github_get(
        f"https://api.github.com/users/{username}/subscriptions",
        params={"per_page": 100},
    )
    return data or []


def get_repo_languages(owner, repo_name):
    """
    Retorna dicionário {linguagem: bytes} de um repositório específico.
    Ex: {"Python": 14023, "Shell": 512}
    """
    data = github_get(f"https://api.github.com/repos/{owner}/{repo_name}/languages")
    return data or {}


def account_age_days(created_at_str):
    """Calcula idade da conta em dias."""
    if not created_at_str:
        return None
    created = datetime.strptime(created_at_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - created).days


def parse_events(events):
    """Extrai contagens de tipos de evento."""
    counts = {
        "push_events": 0,
        "pr_events": 0,
        "issue_events": 0,
        "fork_events": 0,
        "watch_events": 0,
        "create_events": 0,
        "comment_events": 0,
    }
    for event in events:
        t = event.get("type", "")
        if t == "PushEvent":
            counts["push_events"] += 1
        elif t == "PullRequestEvent":
            counts["pr_events"] += 1
        elif t == "IssuesEvent":
            counts["issue_events"] += 1
        elif t == "ForkEvent":
            counts["fork_events"] += 1
        elif t == "WatchEvent":
            counts["watch_events"] += 1
        elif t == "CreateEvent":
            counts["create_events"] += 1
        elif t in ("IssueCommentEvent", "CommitCommentEvent", "PullRequestReviewCommentEvent"):
            counts["comment_events"] += 1
    return counts


# =========================================================
# COLETA DE USUÁRIOS (com checkpoint)
# =========================================================

print("\nColetando lista de usuários...")

checkpoint_users = load_checkpoint("all_users")

if checkpoint_users:
    all_users = checkpoint_users
else:
    all_users = set()
    try:
        for seed in SEED_USERS:
            print(f"  Seed: {seed}")
            followers = get_followers(seed, FOLLOWERS_PER_SEED)
            for f in followers:
                all_users.add(f)
    except RateLimitException:
        pass
    finally:
        all_users = list(all_users)
        save_checkpoint(all_users, "all_users")

all_users = list(all_users)
print(f"\nTotal de usuários coletados: {len(all_users)}")


# =========================================================
# FEATURES DOS USUÁRIOS (com checkpoint)
# =========================================================

print("\nColetando features dos usuários...")

checkpoint_udata = load_checkpoint("users_data")
users_data = checkpoint_udata or []
already_collected = {u["username"] for u in users_data}

try:
    for username in all_users:
        if username in already_collected:
            continue

        print(f"  Usuário: {username}")

        user = get_user(username)
        if user is None:
            continue

        repos = get_user_repos(username)
        events        = get_user_events(username)
        starred       = get_user_starred(username)
        subscriptions = get_user_subscriptions(username)

        # Estatísticas gerais: percorre todos os repos
        language_bytes    = {}
        total_stars       = 0
        total_forks_owned = 0
        total_watchers    = 0
        total_open_issues = 0
        has_readme_profile = 0

        for repo in repos:
            total_stars       += repo.get("stargazers_count", 0)
            total_forks_owned += repo.get("forks_count", 0)
            total_watchers    += repo.get("watchers_count", 0)
            total_open_issues += repo.get("open_issues_count", 0)

            if repo.get("name", "").lower() == username.lower():
                has_readme_profile = 1

        # Linguagens: chama API /languages apenas nos N repos mais recentes
        # (repos já vêm ordenados por updated_at, então os primeiros são os mais ativos)
        for repo in repos:  # já limitado a MAX_REPOS na requisição
            repo_langs = get_repo_languages(username, repo.get("name", ""))
            for lang, bytes_count in repo_langs.items():
                language_bytes[lang] = language_bytes.get(lang, 0) + bytes_count

        dominant_language = max(language_bytes, key=language_bytes.get) if language_bytes else "Unknown"
        event_counts = parse_events(events)

        row = {
            # Identificação
            "username":     username,
            "account_type": user.get("type", "User"),

            # Social
            "followers":                user.get("followers", 0),
            "following":                user.get("following", 0),
            "follower_following_ratio": user.get("followers", 0) / max(user.get("following", 1), 1),

            # Repositórios
            "public_repos":         user.get("public_repos", 0),
            "public_gists":         user.get("public_gists", 0),
            "total_stars_received": total_stars,
            "total_forks_owned":    total_forks_owned,
            "total_watchers":       total_watchers,
            "total_open_issues":    total_open_issues,

            # Linguagens
            "languages":         "|".join(sorted(language_bytes.keys())),
            "num_languages":     len(language_bytes),
            "dominant_language": dominant_language,

            # Perfil/presença
            "bio_length":         len(user.get("bio") or ""),
            "has_company":        int(bool(user.get("company"))),
            "has_blog":           int(bool(user.get("blog"))),
            "has_twitter":        int(bool(user.get("twitter_username"))),
            "has_readme_profile": has_readme_profile,
            "location":           user.get("location") or "Unknown",
            "hireable":           int(bool(user.get("hireable"))),
            "is_site_admin":      int(bool(user.get("site_admin"))),

            # Atividade recente (~90 últimos eventos)
            **event_counts,
            "total_recent_events": len(events),

            # Engajamento externo
            "starred_repos_count":  len(starred),
            "watching_repos_count": len(subscriptions),

            # Tempo de conta
            "account_age_days": account_age_days(user.get("created_at")),
            "last_updated":     user.get("updated_at", ""),
        }

        users_data.append(row)

        # Checkpoint a cada 10 usuários
        if len(users_data) % 10 == 0:
            save_checkpoint(users_data, "users_data")

except RateLimitException:
    print("\n[RATE LIMIT] Salvando progresso dos usuários...")
finally:
    save_checkpoint(users_data, "users_data")

users_df = pd.DataFrame(users_data)
users_df.to_csv("data/raw/users.csv", index=False)

print(f"\nArquivo salvo: data/raw/users.csv")
print(f"Usuários: {len(users_df)}")
print(f"Atributos ({len(users_df.columns)}): {list(users_df.columns)}")