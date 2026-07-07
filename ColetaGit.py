import os
import json
import time
import base64
import shutil
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timezone

# =========================================================
# CONFIGURAÇÕES — ajuste aqui o tamanho da coleta
# =========================================================
load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github+json",
}

SEED_USERS = ["torvalds", "karpathy", "tensorflow", "pytorch", "huggingface"]

MAX_USERS = 1000  # total de usuários distintos a descobrir/coletar
FOLLOWERS_PER_USER = 25  # quantos seguidores buscar ao expandir cada usuário no BFS
MAX_REPOS_PER_USER = 5  # quantos repositórios coletar por usuário
MAX_CONTRIBUTORS_PER_REPO = (
    30  # quantos contribuidores buscar por repositório (limita repos populares)
)
FETCH_README = True  # se False, pula download de README (coleta mais rápida)

DATA_DIR = "data/raw"
CKPT_DIR = "data/checkpoints"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

STATE_PATH = f"{CKPT_DIR}/crawl_state.json"
USERS_CSV = f"{CKPT_DIR}/users_data.csv"
REPOS_CSV = f"{CKPT_DIR}/repos_data.csv"
EDGES_FOLLOW_CSV = f"{CKPT_DIR}/edges_follow.csv"
EDGES_OWNS_CSV = f"{CKPT_DIR}/edges_owns.csv"
EDGES_CONTRIBUTES_CSV = f"{CKPT_DIR}/edges_contributes.csv"


class RateLimitException(Exception):
    pass


# =========================================================
# HTTP + RATE LIMIT
# =========================================================
def check_rate_limit():
    try:
        r = requests.get(
            "https://api.github.com/rate_limit", headers=HEADERS, timeout=15
        )
    except requests.exceptions.RequestException:
        return None, None
    if r.status_code == 200:
        data = r.json()["rate"]
        return data["remaining"], data["reset"]
    return None, None


def github_get(url, params=None, retries=2):
    """GET com tratamento de rate limit, erros transitórios e respostas vazias."""
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=15)
            break
        except requests.exceptions.RequestException as e:
            print(f"  [aviso] erro de rede em {url}: {e}")
            time.sleep(2)
    else:
        return None

    if r.status_code == 202:
        # GitHub ainda está calculando (comum em /contributors de repos grandes)
        return None
    if r.status_code in (403, 429):
        remaining, reset_at = check_rate_limit()
        if remaining == 0:
            reset_time = (
                datetime.fromtimestamp(reset_at).strftime("%H:%M:%S")
                if reset_at
                else "?"
            )
            print(f"\n[RATE LIMIT] Restantes: 0. Reset às {reset_time}.")
            raise RateLimitException("Rate limit da API GitHub atingido.")
        print(f"  [aviso] {r.status_code} em {url} (não é rate limit). Pulando.")
        return None
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        return None

    time.sleep(0.2)
    return r.json()


def github_get_paginated(url, params=None, limit=100):
    """Pagina até atingir `limit` itens ou os resultados acabarem."""
    params = dict(params or {})
    results = []
    page = 1
    while len(results) < limit:
        params["per_page"] = min(100, limit - len(results))
        params["page"] = page
        data = github_get(url, params=params)
        if not data:
            break
        results.extend(data)
        if len(data) < params["per_page"]:
            break
        page += 1
    return results[:limit]


# =========================================================
# CHECKPOINT
# =========================================================
def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            state = json.load(f)
        print(
            f"[checkpoint] estado carregado: {len(state['visited_users'])} usuários descobertos, "
            f"{len(state['users_done'])} já enriquecidos, fila com {len(state['queue'])}"
        )
        return state
    return {
        "queue": list(SEED_USERS),
        "visited_users": list(SEED_USERS),  # descobertos (na fila ou já processados)
        "users_done": [],  # já tiveram features coletadas
        "repos_done": [],  # repos (full_name) já processados
    }


def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def append_rows(rows, path):
    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", index=False, header=write_header)


def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


# =========================================================
# FUNÇÕES DE COLETA — GitHub API
# =========================================================
def get_followers(username, limit):
    data = github_get_paginated(
        f"https://api.github.com/users/{username}/followers", limit=limit
    )
    return [u["login"] for u in (data or [])]


def get_user(username):
    return github_get(f"https://api.github.com/users/{username}")


def get_user_repos(username, limit):
    return (
        github_get_paginated(
            f"https://api.github.com/users/{username}/repos",
            params={"sort": "stars", "direction": "desc"},
            limit=limit,
        )
        or []
    )


def get_user_events(username, limit=100):
    return (
        github_get_paginated(
            f"https://api.github.com/users/{username}/events/public", limit=limit
        )
        or []
    )


def get_user_starred(username, limit=100):
    return (
        github_get_paginated(
            f"https://api.github.com/users/{username}/starred", limit=limit
        )
        or []
    )


def get_user_subscriptions(username, limit=100):
    return (
        github_get_paginated(
            f"https://api.github.com/users/{username}/subscriptions", limit=limit
        )
        or []
    )


def get_repo_languages(owner, repo_name):
    return (
        github_get(f"https://api.github.com/repos/{owner}/{repo_name}/languages") or {}
    )


def get_repo_contributors(owner, repo_name, limit):
    return (
        github_get_paginated(
            f"https://api.github.com/repos/{owner}/{repo_name}/contributors",
            limit=limit,
        )
        or []
    )


def get_repo_readme(owner, repo_name):
    data = github_get(f"https://api.github.com/repos/{owner}/{repo_name}/readme")
    if not data or "content" not in data:
        return ""
    try:
        return base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
    except Exception:
        return ""


def get_user_profile_readme(username):
    # O README de perfil mora no repositório especial "username/username"
    return get_repo_readme(username, username)


def account_age_days(created_at_str):
    if not created_at_str:
        return None
    created = datetime.strptime(created_at_str, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    return (datetime.now(timezone.utc) - created).days


def parse_events(events):
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
        elif t in (
            "IssueCommentEvent",
            "CommitCommentEvent",
            "PullRequestReviewCommentEvent",
        ):
            counts["comment_events"] += 1
    return counts


# =========================================================
# FASE 1 — DESCOBERTA DE USUÁRIOS (BFS no grafo de seguidores)
# =========================================================
def discover_users(state):
    queue = state["queue"]
    visited = set(state["visited_users"])

    while queue and len(visited) < MAX_USERS:
        current = queue.pop(0)
        print(f"  [grafo] expandindo seguidores de {current}")
        followers = get_followers(current, FOLLOWERS_PER_USER)

        follow_edges = [{"follower": f, "followed": current} for f in followers]
        append_rows(follow_edges, EDGES_FOLLOW_CSV)

        for f in followers:
            if len(visited) >= MAX_USERS:
                break
            if f not in visited:
                visited.add(f)
                state["visited_users"].append(f)
                queue.append(f)

        state["queue"] = queue
        save_state(state)

    state["queue"] = queue
    save_state(state)
    print(f"\n[grafo] descoberta concluída: {len(visited)} usuários no total.")


# =========================================================
# FASE 2 — ENRIQUECIMENTO (perfil + repositórios + arestas possui/contribui)
# =========================================================
def collect_user_and_repos(username, repos_done):
    user = get_user(username)
    if user is None:
        return None, [], [], []

    repos = get_user_repos(username, MAX_REPOS_PER_USER)
    events = get_user_events(username)
    starred = get_user_starred(username)
    subscriptions = get_user_subscriptions(username)
    profile_readme = get_user_profile_readme(username) if FETCH_README else ""

    language_bytes = {}
    total_stars = total_forks_owned = total_watchers = total_open_issues = 0
    repo_rows, owns_edges, contributes_edges = [], [], []

    for repo in repos:
        repo_name = repo.get("name", "")
        full_name = repo.get("full_name", f"{username}/{repo_name}")

        total_stars += repo.get("stargazers_count", 0)
        total_forks_owned += repo.get("forks_count", 0)
        total_watchers += repo.get("watchers_count", 0)
        total_open_issues += repo.get("open_issues_count", 0)

        repo_langs = get_repo_languages(username, repo_name)
        for lang, b in repo_langs.items():
            language_bytes[lang] = language_bytes.get(lang, 0) + b

        readme_text = get_repo_readme(username, repo_name) if FETCH_README else ""
        license_info = repo.get("license") or {}

        owns_edges.append({"user": username, "repo": full_name})

        if full_name not in repos_done:
            repo_rows.append(
                {
                    "repo_id": repo.get("id"),
                    "full_name": full_name,
                    "owner": username,
                    "name": repo_name,
                    "description": repo.get("description") or "",
                    "primary_language": repo.get("language") or "Unknown",
                    "languages": "|".join(sorted(repo_langs.keys())),
                    "stars": repo.get("stargazers_count", 0),
                    "forks": repo.get("forks_count", 0),
                    "watchers": repo.get("watchers_count", 0),
                    "open_issues": repo.get("open_issues_count", 0),
                    "size_kb": repo.get("size", 0),
                    "is_fork": int(repo.get("fork", False)),
                    "archived": int(repo.get("archived", False)),
                    "license": license_info.get("name", "Unknown"),
                    "topics": "|".join(repo.get("topics", []) or []),
                    "created_at": repo.get("created_at", ""),
                    "updated_at": repo.get("updated_at", ""),
                    "pushed_at": repo.get("pushed_at", ""),
                    "has_readme": int(bool(readme_text)),
                    "readme_text": readme_text,
                }
            )
            repos_done.add(full_name)

            # Contribuidores só são buscados na primeira vez que vemos o repo
            contributors = get_repo_contributors(
                username, repo_name, MAX_CONTRIBUTORS_PER_REPO
            )
            for c in contributors:
                login = c.get("login")
                if not login:
                    continue
                contributes_edges.append(
                    {
                        "user": login,
                        "repo": full_name,
                        "contributions": c.get("contributions", 0),
                    }
                )

    dominant_language = (
        max(language_bytes, key=language_bytes.get) if language_bytes else "Unknown"
    )
    event_counts = parse_events(events)

    user_row = {
        "username": username,
        "user_id": user.get("id"),
        "account_type": user.get("type", "User"),
        "name": user.get("name") or "",
        "followers": user.get("followers", 0),
        "following": user.get("following", 0),
        "follower_following_ratio": user.get("followers", 0)
        / max(user.get("following", 1), 1),
        "public_repos": user.get("public_repos", 0),
        "public_gists": user.get("public_gists", 0),
        "total_stars_received": total_stars,
        "total_forks_owned": total_forks_owned,
        "total_watchers": total_watchers,
        "total_open_issues": total_open_issues,
        "languages": "|".join(sorted(language_bytes.keys())),
        "num_languages": len(language_bytes),
        "dominant_language": dominant_language,
        "bio": user.get("bio") or "",
        "bio_length": len(user.get("bio") or ""),
        "has_company": int(bool(user.get("company"))),
        "has_blog": int(bool(user.get("blog"))),
        "has_twitter": int(bool(user.get("twitter_username"))),
        "has_readme_profile": int(bool(profile_readme)),
        "profile_readme_text": profile_readme,
        "location": user.get("location") or "Unknown",
        "hireable": int(bool(user.get("hireable"))),
        "is_site_admin": int(bool(user.get("site_admin"))),
        **event_counts,
        "total_recent_events": len(events),
        "starred_repos_count": len(starred),
        "watching_repos_count": len(subscriptions),
        "account_age_days": account_age_days(user.get("created_at")),
        "created_at": user.get("created_at", ""),
        "last_updated": user.get("updated_at", ""),
    }
    return user_row, repo_rows, owns_edges, contributes_edges


def enrich_users(state):
    repos_done = set(state["repos_done"])
    done = set(state["users_done"])
    pending = [u for u in state["visited_users"] if u not in done]
    print(
        f"\nColetando features de {len(pending)} usuários (de {len(state['visited_users'])} descobertos)..."
    )

    for i, username in enumerate(pending, 1):
        print(f"  [{i}/{len(pending)}] {username}")
        user_row, repo_rows, owns_edges, contributes_edges = collect_user_and_repos(
            username, repos_done
        )

        if user_row is not None:
            append_rows([user_row], USERS_CSV)
            append_rows(repo_rows, REPOS_CSV)
            append_rows(owns_edges, EDGES_OWNS_CSV)
            append_rows(contributes_edges, EDGES_CONTRIBUTES_CSV)

        state["users_done"].append(username)
        state["repos_done"] = list(repos_done)
        save_state(state)


# =========================================================
# FASE 3 — MONTAGEM DO GRAFO
# =========================================================
def build_graph():
    import networkx as nx

    users_df = load_csv(USERS_CSV)
    repos_df = load_csv(REPOS_CSV)
    follow_df = load_csv(EDGES_FOLLOW_CSV)
    owns_df = load_csv(EDGES_OWNS_CSV)
    contrib_df = load_csv(EDGES_CONTRIBUTES_CSV)

    G = nx.DiGraph()

    for _, row in users_df.iterrows():
        G.add_node(
            row["username"],
            node_type="user",
            followers=row.get("followers", 0),
            following=row.get("following", 0),
            total_stars_received=row.get("total_stars_received", 0),
            dominant_language=row.get("dominant_language", "Unknown"),
        )
    for _, row in repos_df.iterrows():
        G.add_node(
            row["full_name"],
            node_type="repo",
            stars=row.get("stars", 0),
            forks=row.get("forks", 0),
            primary_language=row.get("primary_language", "Unknown"),
        )
    for _, row in follow_df.iterrows():
        G.add_edge(row["follower"], row["followed"], relation="segue")
    for _, row in owns_df.iterrows():
        G.add_edge(row["user"], row["repo"], relation="possui")
    for _, row in contrib_df.iterrows():
        G.add_edge(
            row["user"],
            row["repo"],
            relation="contribui",
            contributions=row.get("contributions", 0),
        )

    print(f"\n[grafo] {G.number_of_nodes()} nós, {G.number_of_edges()} arestas")
    nx.write_graphml(G, f"{DATA_DIR}/graph.graphml")
    print(f"[grafo] salvo em {DATA_DIR}/graph.graphml")
    return G


def export_final_tables():
    mapping = {
        USERS_CSV: f"{DATA_DIR}/users.csv",
        REPOS_CSV: f"{DATA_DIR}/repos.csv",
        EDGES_FOLLOW_CSV: f"{DATA_DIR}/edges_follow.csv",
        EDGES_OWNS_CSV: f"{DATA_DIR}/edges_owns.csv",
        EDGES_CONTRIBUTES_CSV: f"{DATA_DIR}/edges_contributes.csv",
    }
    for src, dst in mapping.items():
        if os.path.exists(src):
            shutil.copy(src, dst)
    print(f"\nTabelas finais exportadas para {DATA_DIR}/")


def main():
    state = load_state()
    try:
        print("=== Etapa 1: descoberta de usuários (BFS no grafo de seguidores) ===")
        discover_users(state)

        print("\n=== Etapa 2: coleta de features de usuários e repositórios ===")
        enrich_users(state)

    except RateLimitException:
        print(
            "\n[RATE LIMIT] Progresso salvo em checkpoints. Rode o script novamente para continuar."
        )
        return

    export_final_tables()

    print("\n=== Etapa 3: construção do grafo ===")
    build_graph()

    users_df = load_csv(USERS_CSV)
    repos_df = load_csv(REPOS_CSV)
    print(f"\nConcluído. Usuários: {len(users_df)} | Repositórios: {len(repos_df)}")


if __name__ == "__main__":
    main()
