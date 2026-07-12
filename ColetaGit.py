"""
Coleta usuários, repositórios e grafo de relações da API do GitHub.

Melhorias de velocidade em relação à versão anterior:
  - Chamadas independentes de cada usuário (eventos, starred, subscriptions,
    repos, README de perfil) rodam em paralelo via ThreadPoolExecutor.
  - Chamadas por repositório (linguagens, README, contribuidores) também
    rodam em paralelo, uma por repositório ao mesmo tempo.
  - Um threading.Event global interrompe todas as threads ao primeiro
    RateLimitException, garantindo que o checkpoint seja salvo.

Requisitos: pip install requests pandas python-dotenv networkx
"""

import os, json, time, base64, shutil, threading, requests, pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github+json"}

SEED_USERS = ["torvalds", "karpathy", "tensorflow", "pytorch", "huggingface"]
MAX_USERS = 5000
FOLLOWERS_PER_USER = 25
MAX_REPOS_PER_USER = 5
MAX_CONTRIBUTORS_PER_REPO = 30
FETCH_README = True
PARALLEL_WORKERS = 4  # threads simultâneas por usuário
REQUEST_DELAY = 0.15  # segundos de cortesia entre requisições

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

# Flag compartilhada entre threads: sinaliza quando rate limit é atingido
_stop_flag = threading.Event()


class RateLimitException(Exception):
    pass


# =========================================================
# HTTP
# =========================================================
def check_rate_limit():
    try:
        r = requests.get(
            "https://api.github.com/rate_limit", headers=HEADERS, timeout=15
        )
        if r.status_code == 200:
            d = r.json()["rate"]
            return d["remaining"], d["reset"]
    except Exception:
        pass
    return None, None


def github_get(url, params=None, retries=2):
    if _stop_flag.is_set():
        return None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=15)
            break
        except requests.exceptions.RequestException as e:
            if attempt == retries:
                return None
            time.sleep(1)
    else:
        return None

    if r.status_code == 202:
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
            _stop_flag.set()
            raise RateLimitException("Rate limit atingido.")
        return None
    if r.status_code in (404, 204):
        return None
    if r.status_code != 200:
        return None

    time.sleep(REQUEST_DELAY)
    return r.json()


def github_get_paginated(url, params=None, limit=100):
    params = dict(params or {})
    results, page = [], 1
    while len(results) < limit:
        if _stop_flag.is_set():
            break
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
            f"[checkpoint] {len(state['visited_users'])} descobertos, "
            f"{len(state['users_done'])} enriquecidos, fila: {len(state['queue'])}"
        )
        return state
    return {
        "queue": list(SEED_USERS),
        "visited_users": list(SEED_USERS),
        "users_done": [],
        "repos_done": [],
    }


def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def append_rows(rows, path):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", index=False, header=not os.path.exists(path))


def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


# =========================================================
# FUNÇÕES DE COLETA
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
    return get_repo_readme(username, username)


# =========================================================
# UTILITÁRIOS
# =========================================================
def account_age_days(created_at_str):
    if not created_at_str:
        return None
    try:
        created = datetime.strptime(created_at_str, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        return (datetime.now(timezone.utc) - created).days
    except ValueError:
        return None


def parse_events(events):
    counts = {
        k: 0
        for k in [
            "push_events",
            "pr_events",
            "issue_events",
            "fork_events",
            "watch_events",
            "create_events",
            "comment_events",
        ]
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
# FASE 1 — DESCOBERTA (BFS)
# =========================================================
def discover_users(state):
    queue = state["queue"]
    visited = set(state["visited_users"])

    while queue and len(visited) < MAX_USERS and not _stop_flag.is_set():
        current = queue.pop(0)
        print(f"  [BFS] expandindo {current} ({len(visited)}/{MAX_USERS})")
        followers = get_followers(current, FOLLOWERS_PER_USER)

        append_rows(
            [{"follower": f, "followed": current} for f in followers], EDGES_FOLLOW_CSV
        )

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
    print(f"[BFS] {len(visited)} usuários descobertos.")


# =========================================================
# FASE 2 — ENRIQUECIMENTO (paralelo por usuário)
# =========================================================
def _fetch_repo_data(username, repo, repos_done):
    """Busca linguagens, README e contribuidores de um repositório em paralelo."""
    repo_name = repo.get("name", "")
    full_name = repo.get("full_name", f"{username}/{repo_name}")
    if _stop_flag.is_set():
        return None, None

    langs = get_repo_languages(username, repo_name)
    readme = get_repo_readme(username, repo_name) if FETCH_README else ""
    contrib_edges = []

    if full_name not in repos_done:
        contributors = get_repo_contributors(
            username, repo_name, MAX_CONTRIBUTORS_PER_REPO
        )
        for c in contributors:
            login = c.get("login")
            if login:
                contrib_edges.append(
                    {
                        "user": login,
                        "repo": full_name,
                        "contributions": c.get("contributions", 0),
                    }
                )
    return langs, readme, contrib_edges, full_name


def collect_user_and_repos(username, repos_done):
    """Coleta dados de um usuário, paralelizando chamadas independentes."""
    user = get_user(username)
    if user is None or _stop_flag.is_set():
        return None, [], [], []

    # — Chamadas de nível usuário em paralelo —
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        f_repos = pool.submit(get_user_repos, username, MAX_REPOS_PER_USER)
        f_events = pool.submit(get_user_events, username)
        f_starred = pool.submit(get_user_starred, username)
        f_subs = pool.submit(get_user_subscriptions, username)
        f_readme = (
            pool.submit(get_user_profile_readme, username) if FETCH_README else None
        )

        repos = f_repos.result() or []
        events = f_events.result() or []
        starred = f_starred.result() or []
        subscriptions = f_subs.result() or []
        profile_readme = f_readme.result() if f_readme else ""

    if _stop_flag.is_set():
        return None, [], [], []

    # — Chamadas de nível repositório em paralelo —
    language_bytes = {}
    total_stars = total_forks_owned = total_watchers = total_open_issues = 0
    repo_rows, owns_edges, contributes_edges = [], [], []

    futures = {}
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        for repo in repos:
            fut = pool.submit(_fetch_repo_data, username, repo, repos_done)
            futures[fut] = repo

        for fut in as_completed(futures):
            if _stop_flag.is_set():
                break
            repo = futures[fut]
            try:
                result = fut.result()
            except RateLimitException:
                break
            if result is None or result[0] is None:
                continue

            langs, readme, contrib_edges_repo, full_name = result
            repo_name = repo.get("name", "")
            license_info = repo.get("license") or {}

            total_stars += repo.get("stargazers_count", 0)
            total_forks_owned += repo.get("forks_count", 0)
            total_watchers += repo.get("watchers_count", 0)
            total_open_issues += repo.get("open_issues_count", 0)

            for lang, b in langs.items():
                language_bytes[lang] = language_bytes.get(lang, 0) + b

            owns_edges.append({"user": username, "repo": full_name})
            contributes_edges.extend(contrib_edges_repo)

            if full_name not in repos_done:
                repo_rows.append(
                    {
                        "repo_id": repo.get("id"),
                        "full_name": full_name,
                        "owner": username,
                        "name": repo_name,
                        "description": repo.get("description") or "",
                        "primary_language": repo.get("language") or "Unknown",
                        "languages": "|".join(sorted(langs.keys())),
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
                        "has_readme": int(bool(readme)),
                        "readme_text": readme,
                    }
                )
                repos_done.add(full_name)

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
    print(f"\nEnriquecendo {len(pending)} usuários...")

    for i, username in enumerate(pending, 1):
        if _stop_flag.is_set():
            break
        print(f"  [{i}/{len(pending)}] {username}")
        user_row, repo_rows, owns_edges, contrib_edges = collect_user_and_repos(
            username, repos_done
        )

        if user_row is not None:
            append_rows([user_row], USERS_CSV)
            append_rows(repo_rows, REPOS_CSV)
            append_rows(owns_edges, EDGES_OWNS_CSV)
            append_rows(contrib_edges, EDGES_CONTRIBUTES_CSV)

        state["users_done"].append(username)
        state["repos_done"] = list(repos_done)
        save_state(state)


# =========================================================
# FASE 3 — GRAFO
# =========================================================
def build_graph():
    import networkx as nx

    users_df = load_csv(USERS_CSV)
    repos_df = load_csv(REPOS_CSV)
    follow_df = load_csv(EDGES_FOLLOW_CSV)
    owns_df = load_csv(EDGES_OWNS_CSV)
    contrib_df = load_csv(EDGES_CONTRIBUTES_CSV)

    G = nx.DiGraph()
    for _, r in users_df.iterrows():
        G.add_node(
            r["username"],
            node_type="user",
            followers=r.get("followers", 0),
            dominant_language=r.get("dominant_language", "Unknown"),
        )
    for _, r in repos_df.iterrows():
        G.add_node(
            r["full_name"],
            node_type="repo",
            stars=r.get("stars", 0),
            primary_language=r.get("primary_language", "Unknown"),
        )
    for _, r in follow_df.iterrows():
        G.add_edge(r["follower"], r["followed"], relation="segue")
    for _, r in owns_df.iterrows():
        G.add_edge(r["user"], r["repo"], relation="possui")
    for _, r in contrib_df.iterrows():
        G.add_edge(
            r["user"],
            r["repo"],
            relation="contribui",
            contributions=r.get("contributions", 0),
        )

    print(f"[grafo] {G.number_of_nodes()} nós, {G.number_of_edges()} arestas")
    nx.write_graphml(G, f"{DATA_DIR}/graph.graphml")
    print(f"[grafo] salvo em {DATA_DIR}/graph.graphml")
    return G


def export_final_tables():
    for src, dst in {
        USERS_CSV: f"{DATA_DIR}/users_data.csv",
        REPOS_CSV: f"{DATA_DIR}/repos_data.csv",
        EDGES_FOLLOW_CSV: f"{DATA_DIR}/edges_follow.csv",
        EDGES_OWNS_CSV: f"{DATA_DIR}/edges_owns.csv",
        EDGES_CONTRIBUTES_CSV: f"{DATA_DIR}/edges_contributes.csv",
    }.items():
        if os.path.exists(src):
            shutil.copy(src, dst)
    print(f"Tabelas exportadas para {DATA_DIR}/")


# =========================================================
# MAIN
# =========================================================
def main():
    state = load_state()
    try:
        print("=== Etapa 1: descoberta de usuários (BFS) ===")
        discover_users(state)

        if _stop_flag.is_set():
            raise RateLimitException()

        print("\n=== Etapa 2: enriquecimento ===")
        enrich_users(state)

    except RateLimitException:
        print("\n[RATE LIMIT] Progresso salvo. Execute novamente para continuar.")
        return

    if _stop_flag.is_set():
        print("\n[RATE LIMIT] Progresso salvo. Execute novamente para continuar.")
        return

    export_final_tables()
    print("\n=== Etapa 3: construção do grafo ===")
    build_graph()
    print(
        f"\nConcluído. Usuários: {len(load_csv(USERS_CSV))} | "
        f"Repositórios: {len(load_csv(REPOS_CSV))}"
    )


if __name__ == "__main__":
    main()
