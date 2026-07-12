"""
Recomendador de repositórios GitHub.

Novidade: aceita qualquer usuário e qualquer repositório — se não estiverem
na base local, os dados são buscados ao vivo na API do GitHub antes de
calcular as features. Isso permite usar o recomendador sem precisar ter
coletado o usuário/repo previamente no ColetaGit.py.

Uso:
    # usuários e repos da base local
    python Recomendar.py --usuario torvalds --top_k 10
    python Recomendar.py --usuario torvalds --top_k 10 --modelo data/model/contrib_model.joblib

    # qualquer usuário/repo (busca ao vivo se necessário)
    python Recomendar.py --usuario gvanrossum --repos "django/django,pallets/flask,psf/requests"
"""

import os, time, base64, requests
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import joblib
from dotenv import load_dotenv
from datetime import datetime, timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
_GH_HEADERS = {
    "Authorization": f"token {os.getenv('GITHUB_TOKEN','')}" ,
    "Accept": "application/vnd.github+json",
}

DATA_DIR  = "data/raw"
USERS_CSV = f"{DATA_DIR}/users_data.csv"
REPOS_CSV = f"{DATA_DIR}/repos_data.csv"

# =========================================================
# FEATURES
# =========================================================
FEATURE_COLUMNS = [
    "text_similarity",
    "language_match",
    "follows_owner",
    "common_contributors_following",
    "shared_repos_with_owner",
    "graph_proximity",
    "repo_popularity",
    "repo_open_issues",
    "repo_freshness",
    "repo_size_kb",
    "repo_is_fork",
    "repo_is_archived",
    "repo_contributor_count",
    "user_pr_ratio",
    "user_activity_score",
    "user_external_engagement",
    "user_influence",
    "user_account_age_days",
    "user_num_languages",
    "user_public_repos",
    "user_total_stars",
    "user_followers",
    "user_following",
    "user_starred_repos_count",
    "user_watching_repos_count",
    "user_has_readme_profile",
    "user_bio_length",
    "user_has_company",
    "user_has_blog",
    "repo_stars",
    "repo_forks",
    "repo_watchers",
    "repo_has_readme",
    "owner_follower_count",
]

WEIGHTS = {f: 1.0 / len(FEATURE_COLUMNS) for f in FEATURE_COLUMNS}
# Ajustes manuais nos sinais mais informativos
WEIGHTS.update({
    "text_similarity":               0.10,
    "language_match":                0.08,
    "follows_owner":                 0.10,
    "common_contributors_following": 0.10,
    "shared_repos_with_owner":       0.07,
    "graph_proximity":               0.10,
    "repo_popularity":               0.06,
    "repo_freshness":                0.06,
    "user_pr_ratio":                 0.05,
    "user_activity_score":           0.04,
    "user_influence":                0.04,
    "owner_follower_count":          0.04,
})
# Normaliza para somar 1
_total = sum(WEIGHTS.values())
WEIGHTS = {k: v / _total for k, v in WEIGHTS.items()}


# =========================================================
# UTILITÁRIOS
# =========================================================
def _minmax(v):
    v = np.asarray(v, dtype=float)
    lo, hi = v.min(), v.max()
    return np.full_like(v, 0.5) if hi - lo < 1e-9 else (v - lo) / (hi - lo)

def _split_set(text):
    if not isinstance(text, str) or not text:
        return set()
    return {t.strip() for t in text.split("|") if t.strip()}

def _jaccard(a, b):
    u = a | b
    return 0.0 if not u else len(a & b) / len(u)

def _safe(val, default=0.0):
    try:
        v = float(val)
        return default if (v != v) else v   # NaN check
    except (TypeError, ValueError):
        return default

def _days_since(date_str):
    if not isinstance(date_str, str) or not date_str:
        return 9999.0
    try:
        dt = datetime.fromisoformat(date_str.replace("Z","+00:00"))
        return max(0.0, (datetime.now(timezone.utc) - dt).days)
    except ValueError:
        return 9999.0


# =========================================================
# CLIENTE GITHUB (para busca ao vivo)
# =========================================================
def _gh_get(url, params=None):
    """GET simples na API GitHub; retorna None em caso de erro."""
    try:
        r = requests.get(url, headers=_GH_HEADERS, params=params, timeout=15)
        time.sleep(0.2)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def _gh_paginated(url, params=None, limit=50):
    params = dict(params or {})
    results, page = [], 1
    while len(results) < limit:
        params["per_page"] = min(100, limit - len(results))
        params["page"]     = page
        data = _gh_get(url, params)
        if not data:
            break
        results.extend(data)
        if len(data) < params["per_page"]:
            break
        page += 1
    return results[:limit]

def _gh_readme(owner, repo):
    data = _gh_get(f"https://api.github.com/repos/{owner}/{repo}/readme")
    if not data or "content" not in data:
        return ""
    try:
        return base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
    except Exception:
        return ""

def _account_age_days(s):
    if not s:
        return 0.0
    try:
        dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return float((datetime.now(timezone.utc) - dt).days)
    except ValueError:
        return 0.0

def _parse_events(events):
    c = {k: 0 for k in ["push_events","pr_events","issue_events","comment_events"]}
    for e in events:
        t = e.get("type","")
        if   t == "PushEvent":        c["push_events"]   += 1
        elif t == "PullRequestEvent": c["pr_events"]     += 1
        elif t == "IssuesEvent":      c["issue_events"]  += 1
        elif t in ("IssueCommentEvent","CommitCommentEvent","PullRequestReviewCommentEvent"):
            c["comment_events"] += 1
    return c


# =========================================================
# BUSCA AO VIVO: monta um "user_row" e "repo_row" do zero via API
# =========================================================
def _fetch_user_live(username):
    """Retorna uma linha no mesmo formato do users_data.csv, via API."""
    user = _gh_get(f"https://api.github.com/users/{username}")
    if not user:
        return None

    repos      = _gh_paginated(f"https://api.github.com/users/{username}/repos",
                                {"sort":"stars","direction":"desc"}, limit=5)
    events     = _gh_paginated(f"https://api.github.com/users/{username}/events/public", limit=50)
    starred    = _gh_paginated(f"https://api.github.com/users/{username}/starred", limit=50)
    subs       = _gh_paginated(f"https://api.github.com/users/{username}/subscriptions", limit=50)
    readme     = _gh_readme(username, username)   # README de perfil

    lang_bytes = {}
    for repo in repos:
        rname = repo.get("name","")
        langs = _gh_get(f"https://api.github.com/repos/{username}/{rname}/languages") or {}
        for lang, b in langs.items():
            lang_bytes[lang] = lang_bytes.get(lang, 0) + b

    ecounts = _parse_events(events)
    total_ev = max(sum(ecounts.values()), 1)
    dom_lang = max(lang_bytes, key=lang_bytes.get) if lang_bytes else "Unknown"
    followers = user.get("followers",0)
    following = user.get("following",0)

    return {
        "username":            username,
        "user_id":             user.get("id"),
        "account_type":        user.get("type","User"),
        "name":                user.get("name") or "",
        "followers":           followers,
        "following":           following,
        "follower_following_ratio": followers / max(following,1),
        "public_repos":        user.get("public_repos",0),
        "public_gists":        user.get("public_gists",0),
        "total_stars_received":sum(r.get("stargazers_count",0) for r in repos),
        "total_forks_owned":   sum(r.get("forks_count",0)      for r in repos),
        "total_watchers":      sum(r.get("watchers_count",0)   for r in repos),
        "total_open_issues":   sum(r.get("open_issues_count",0)for r in repos),
        "languages":           "|".join(sorted(lang_bytes.keys())),
        "num_languages":       len(lang_bytes),
        "dominant_language":   dom_lang,
        "bio":                 user.get("bio") or "",
        "bio_length":          len(user.get("bio") or ""),
        "has_company":         int(bool(user.get("company"))),
        "has_blog":            int(bool(user.get("blog"))),
        "has_twitter":         int(bool(user.get("twitter_username"))),
        "has_readme_profile":  int(bool(readme)),
        "profile_readme_text": readme,
        "location":            user.get("location") or "Unknown",
        "hireable":            int(bool(user.get("hireable"))),
        "is_site_admin":       int(bool(user.get("site_admin"))),
        **ecounts,
        "total_recent_events":  len(events),
        "starred_repos_count":  len(starred),
        "watching_repos_count": len(subs),
        "account_age_days":     _account_age_days(user.get("created_at","")),
        "created_at":           user.get("created_at",""),
        "last_updated":         user.get("updated_at",""),
    }


def _fetch_repo_live(full_name):
    """Retorna uma linha no mesmo formato do repos_data.csv, via API."""
    owner, repo_name = full_name.split("/", 1) if "/" in full_name else (full_name, full_name)
    repo = _gh_get(f"https://api.github.com/repos/{owner}/{repo_name}")
    if not repo:
        return None, []

    langs       = _gh_get(f"https://api.github.com/repos/{owner}/{repo_name}/languages") or {}
    readme_text = _gh_readme(owner, repo_name)
    contributors= _gh_paginated(f"https://api.github.com/repos/{owner}/{repo_name}/contributors", limit=30)
    license_info = repo.get("license") or {}

    contrib_edges = [
        {"user": c.get("login"), "repo": full_name, "contributions": c.get("contributions",0)}
        for c in contributors if c.get("login")
    ]

    row = {
        "repo_id":          repo.get("id"),
        "full_name":        full_name,
        "owner":            owner,
        "name":             repo_name,
        "description":      repo.get("description") or "",
        "primary_language": repo.get("language") or "Unknown",
        "languages":        "|".join(sorted(langs.keys())),
        "stars":            repo.get("stargazers_count",0),
        "forks":            repo.get("forks_count",0),
        "watchers":         repo.get("watchers_count",0),
        "open_issues":      repo.get("open_issues_count",0),
        "size_kb":          repo.get("size",0),
        "is_fork":          int(repo.get("fork",False)),
        "archived":         int(repo.get("archived",False)),
        "license":          license_info.get("name","Unknown"),
        "topics":           "|".join(repo.get("topics",[]) or []),
        "created_at":       repo.get("created_at",""),
        "updated_at":       repo.get("updated_at",""),
        "pushed_at":        repo.get("pushed_at",""),
        "has_readme":       int(bool(readme_text)),
        "readme_text":      readme_text,
    }
    return row, contrib_edges


def _fetch_following_live(username, limit=100):
    """Lista de quem o usuário segue (para features sociais)."""
    data = _gh_paginated(f"https://api.github.com/users/{username}/following", limit=limit)
    return [u["login"] for u in (data or [])]


# =========================================================
# RECOMENDADOR
# =========================================================
class Recommender:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        users_path = f"{data_dir}/users_data.csv"
        repos_path = f"{data_dir}/repos_data.csv"

        self.users_df = (pd.read_csv(users_path)
                         .drop_duplicates("username")
                         .set_index("username", drop=False)
                         if os.path.exists(users_path) else pd.DataFrame())
        self.repos_df = (pd.read_csv(repos_path)
                         .drop_duplicates("full_name")
                         .set_index("full_name", drop=False)
                         if os.path.exists(repos_path) else pd.DataFrame())

        self.follow_df  = self._read_or_empty(f"{data_dir}/edges_follow.csv",       ["follower","followed"])
        self.owns_df    = self._read_or_empty(f"{data_dir}/edges_owns.csv",          ["user","repo"])
        self.contrib_df = self._read_or_empty(f"{data_dir}/edges_contributes.csv",   ["user","repo","contributions"])

        self._build_graph()
        self._build_lookup_sets()
        self._fit_tfidf()

        # cache de vetores TF-IDF para itens buscados ao vivo
        self._user_vec_cache = {}
        self._repo_vec_cache = {}

    @staticmethod
    def _read_or_empty(path, cols):
        return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=cols)

    # ------ grafo ------
    def _build_graph(self):
        G = nx.MultiDiGraph()
        for u in self.users_df.get("username", pd.Series([], dtype=str)):
            G.add_node(u, node_type="user")
        for r in self.repos_df.get("full_name", pd.Series([], dtype=str)):
            G.add_node(r, node_type="repo")
        for _, row in self.follow_df.iterrows():
            G.add_edge(row["follower"], row["followed"], relation="segue")
        for _, row in self.owns_df.iterrows():
            G.add_edge(row["user"], row["repo"], relation="possui")
        for _, row in self.contrib_df.iterrows():
            G.add_edge(row["user"], row["repo"], relation="contribui")
        self.G = G
        self.G_und = G.to_undirected()

    # ------ lookup sets ------
    def _build_lookup_sets(self):
        self.following = (self.follow_df.groupby("follower")["followed"].apply(set).to_dict()
                          if not self.follow_df.empty else {})
        self.contributors_of = (self.contrib_df.groupby("repo")["user"].apply(set).to_dict()
                                 if not self.contrib_df.empty else {})
        c = (self.contrib_df.groupby("user")["repo"].apply(set).to_dict()
             if not self.contrib_df.empty else {})
        o = (self.owns_df.groupby("user")["repo"].apply(set).to_dict()
             if not self.owns_df.empty else {})
        self.user_repos = {u: c.get(u,set()) | o.get(u,set()) for u in set(c)|set(o)}

    # ------ TF-IDF ------
    def _user_text(self, row):
        return " ".join(filter(None, [
            str(row.get("profile_readme_text") or ""),
            str(row.get("bio") or ""),
        ]))

    def _repo_text(self, row):
        return " ".join(filter(None, [
            str(row.get("readme_text")  or ""),
            str(row.get("description") or ""),
            str(row.get("topics")      or "").replace("|"," "),
        ]))

    def _fit_tfidf(self):
        utexts = ([self._user_text(r) for _, r in self.users_df.iterrows()]
                  if not self.users_df.empty else [])
        rtexts = ([self._repo_text(r) for _, r in self.repos_df.iterrows()]
                  if not self.repos_df.empty else [])
        corpus = utexts + rtexts
        if not any(t.strip() for t in corpus):
            corpus = ["sem conteudo"]
        self.vectorizer = TfidfVectorizer(max_features=20000, sublinear_tf=True)
        self.vectorizer.fit(corpus)
        self._uvec = self.vectorizer.transform(utexts) if utexts else None
        self._rvec = self.vectorizer.transform(rtexts) if rtexts else None
        self._uidx = {u: i for i, u in enumerate(self.users_df.get("username", []))}
        self._ridx = {r: i for i, r in enumerate(self.repos_df.get("full_name", []))}

    def _get_user_vec(self, username):
        if username in self._user_vec_cache:
            return self._user_vec_cache[username]
        i = self._uidx.get(username)
        if i is not None and self._uvec is not None:
            return self._uvec[i]
        return None

    def _get_repo_vec(self, repo):
        if repo in self._repo_vec_cache:
            return self._repo_vec_cache[repo]
        i = self._ridx.get(repo)
        if i is not None and self._rvec is not None:
            return self._rvec[i]
        return None

    # ------ busca ao vivo ------
    def _ensure_user(self, username):
        """Se o usuario nao esta na base, busca ao vivo via API e reconstroi
        todas as conexoes no grafo para que as features sociais funcionem.

        Chamadas feitas:
          - perfil do usuario (1 call via _fetch_user_live, que ja inclui repos e eventos)
          - lista de following (1 call) -> arestas "segue" no grafo
          - lista de repositorios proprios (1 call) -> arestas "possui" no grafo
        """
        if username in (self.users_df.index if not self.users_df.empty else []):
            return
        print(f"  [live] buscando perfil de '{username}'...")
        row = _fetch_user_live(username)
        if row is None:
            print(f"  [aviso] usuario '{username}' nao encontrado na API.")
            return

        # 1. Adiciona a tabela em memoria
        new_df = pd.DataFrame([row]).set_index("username", drop=False)
        self.users_df = (pd.concat([self.users_df, new_df])
                         if not self.users_df.empty else new_df)

        # 2. Vetor TF-IDF (usa o vectorizer ja fittado, sem refittar)
        vec = self.vectorizer.transform([self._user_text(row)])
        self._user_vec_cache[username] = vec
        self._uidx[username] = -1

        # 3. No no grafo
        if not self.G.has_node(username):
            self.G.add_node(username, node_type="user")
            self.G_und.add_node(username, node_type="user")

        # 4. Following -> arestas "segue"
        # Habilita: follows_owner, common_contributors_following, graph_proximity
        print(f"  [live] buscando following de '{username}'...")
        following_list = _fetch_following_live(username, limit=100)
        if following_list:
            self.following[username] = set(following_list)
            for followed in following_list:
                # So conecta a nos que ja existem no grafo (da base local)
                if self.G.has_node(followed):
                    if not self.G.has_edge(username, followed):
                        self.G.add_edge(username, followed, relation="segue")
                    if not self.G_und.has_edge(username, followed):
                        self.G_und.add_edge(username, followed, relation="segue")

        # 5. Repositorios proprios -> arestas "possui"
        # Habilita: shared_repos_with_owner
        print(f"  [live] buscando repositorios de '{username}'...")
        repos_live = _gh_paginated(
            f"https://api.github.com/users/{username}/repos",
            {"sort": "stars", "direction": "desc"}, limit=10)
        user_repo_set = set()
        for repo in repos_live:
            fn = repo.get("full_name", "")
            if not fn:
                continue
            user_repo_set.add(fn)
            if not self.G.has_node(fn):
                self.G.add_node(fn, node_type="repo")
                self.G_und.add_node(fn, node_type="repo")
            if not self.G.has_edge(username, fn):
                self.G.add_edge(username, fn, relation="possui")
            if not self.G_und.has_edge(username, fn):
                self.G_und.add_edge(username, fn, relation="possui")
        if user_repo_set:
            self.user_repos[username] = self.user_repos.get(username, set()) | user_repo_set

    def _ensure_repo(self, full_name):
        """Se o repo não está na base, busca ao vivo e adiciona em memória."""
        if full_name in (self.repos_df.index if not self.repos_df.empty else []):
            return
        print(f"  [live] buscando repositório '{full_name}' na API...")
        row, contrib_edges = _fetch_repo_live(full_name)
        if row is None:
            print(f"  [aviso] repositório '{full_name}' não encontrado na API.")
            return

        new_df = pd.DataFrame([row]).set_index("full_name", drop=False)
        self.repos_df = (pd.concat([self.repos_df, new_df])
                         if not self.repos_df.empty else new_df)

        vec = self.vectorizer.transform([self._repo_text(row)])
        self._repo_vec_cache[full_name] = vec
        self._ridx[full_name] = -1

        contributors = {e["user"] for e in contrib_edges}
        self.contributors_of[full_name] = contributors
        for e in contrib_edges:
            u = e["user"]
            self.user_repos.setdefault(u, set()).add(full_name)

        # Adiciona nó ao grafo
        if not self.G.has_node(full_name):
            self.G.add_node(full_name, node_type="repo")
            self.G_und.add_node(full_name, node_type="repo")
        # Aresta "possui"
        owner = row.get("owner","")
        if owner and self.G.has_node(owner):
            self.G.add_edge(owner, full_name, relation="possui")
            self.G_und.add_edge(owner, full_name, relation="possui")

    # ------ features individuais ------
    def _text_sim(self, username, repo):
        uv = self._get_user_vec(username)
        rv = self._get_repo_vec(repo)
        if uv is None or rv is None:
            return 0.0
        return float(cosine_similarity(uv, rv)[0, 0])

    def _lang_match(self, username, repo):
        ul = _split_set(self.users_df.at[username,"languages"] if username in self.users_df.index else "")
        rl = _split_set(self.repos_df.at[repo,"languages"]     if repo     in self.repos_df.index else "")
        return _jaccard(ul, rl)

    def _follows_owner(self, username, repo):
        if repo not in self.repos_df.index:
            return 0.0
        owner = self.repos_df.at[repo,"owner"]
        return 1.0 if owner in self.following.get(username,set()) else 0.0

    def _common_contrib_following(self, username, repo):
        return len(self.contributors_of.get(repo,set()) & self.following.get(username,set()))

    def _shared_repos_owner(self, username, repo):
        if repo not in self.repos_df.index:
            return 0
        owner = self.repos_df.at[repo,"owner"]
        if owner == username:
            return 0
        return len((self.user_repos.get(username,set()) & self.user_repos.get(owner,set())) - {repo})

    @staticmethod
    def _graph_prox(distances, repo):
        d = distances.get(repo)
        return 0.0 if d is None else 1.0/(1.0+d)

    def _repo_feats(self, repo):
        if repo not in self.repos_df.index:
            return {k: 0.0 for k in ["repo_popularity","repo_open_issues","repo_freshness",
                                      "repo_size_kb","repo_is_fork","repo_is_archived",
                                      "repo_contributor_count","repo_stars","repo_forks",
                                      "repo_watchers","repo_has_readme"]}
        r = self.repos_df.loc[repo]
        return {
            "repo_popularity":      float(np.log1p(_safe(r.get("stars")) + _safe(r.get("forks")) + _safe(r.get("watchers")))),
            "repo_open_issues":     float(np.log1p(_safe(r.get("open_issues")))),
            "repo_freshness":       1.0/(1.0+_days_since(str(r.get("pushed_at","")))),
            "repo_size_kb":         float(np.log1p(_safe(r.get("size_kb")))),
            "repo_is_fork":         float(_safe(r.get("is_fork"))),
            "repo_is_archived":     float(_safe(r.get("archived"))),
            "repo_contributor_count": float(len(self.contributors_of.get(repo,set()))),
            "repo_stars":           float(_safe(r.get("stars"))),
            "repo_forks":           float(_safe(r.get("forks"))),
            "repo_watchers":        float(_safe(r.get("watchers"))),
            "repo_has_readme":      float(_safe(r.get("has_readme"))),
        }

    def _user_feats(self, username):
        if username not in self.users_df.index:
            return {k: 0.0 for k in ["user_pr_ratio","user_activity_score",
                                      "user_external_engagement","user_influence",
                                      "user_account_age_days","user_num_languages",
                                      "user_public_repos","user_total_stars",
                                      "user_followers","user_following",
                                      "user_starred_repos_count","user_watching_repos_count",
                                      "user_has_readme_profile","user_bio_length",
                                      "user_has_company","user_has_blog"]}
        u = self.users_df.loc[username]
        total_ev = max(_safe(u.get("total_recent_events")), 1.0)
        pr       = _safe(u.get("pr_events"))
        push     = _safe(u.get("push_events"))
        issue    = _safe(u.get("issue_events"))
        comment  = _safe(u.get("comment_events"))
        fl       = _safe(u.get("followers"))
        fg       = _safe(u.get("following"))
        return {
            "user_pr_ratio":            pr / total_ev,
            "user_activity_score":      float(np.log1p(push+pr+issue+comment)),
            "user_external_engagement": float(np.log1p(_safe(u.get("starred_repos_count")) +
                                                       _safe(u.get("watching_repos_count")))),
            "user_influence":           fl/(fl+fg+1.0),
            "user_account_age_days":    _safe(u.get("account_age_days")),
            "user_num_languages":       _safe(u.get("num_languages")),
            "user_public_repos":        _safe(u.get("public_repos")),
            "user_total_stars":         float(np.log1p(_safe(u.get("total_stars_received")))),
            "user_followers":           fl,
            "user_following":           fg,
            "user_starred_repos_count": _safe(u.get("starred_repos_count")),
            "user_watching_repos_count":_safe(u.get("watching_repos_count")),
            "user_has_readme_profile":  _safe(u.get("has_readme_profile")),
            "user_bio_length":          _safe(u.get("bio_length")),
            "user_has_company":         _safe(u.get("has_company")),
            "user_has_blog":            _safe(u.get("has_blog")),
        }

    def _owner_follower_count(self, repo):
        if repo not in self.repos_df.index:
            return 0.0
        owner = self.repos_df.at[repo,"owner"]
        if owner not in self.users_df.index:
            return 0.0
        return float(np.log1p(_safe(self.users_df.at[owner,"followers"])))

    # ------ API pública ------
    def build_feature_matrix(self, username, candidate_repos, unlink_repos=None):
        # Garante que usuário e todos os repos estão disponíveis (base ou API live)
        self._ensure_user(username)
        for repo in candidate_repos:
            self._ensure_repo(repo)

        if username not in self.users_df.index:
            raise ValueError(f"Usuário '{username}' não encontrado nem na base nem na API.")

        valid_repos = [r for r in candidate_repos if r in self.repos_df.index]
        missing     = sorted(set(candidate_repos) - set(valid_repos))
        if missing:
            print(f"  [aviso] {len(missing)} repo(s) ignorados (não encontrados): {missing[:5]}")

        graph_dist = self.G_und
        if unlink_repos:
            graph_dist = self.G_und.copy()
            for repo in unlink_repos:
                edata = graph_dist.get_edge_data(username, repo) or {}
                for k, attrs in list(edata.items()):
                    if attrs.get("relation") == "contribui":
                        graph_dist.remove_edge(username, repo, k)

        distances = (nx.single_source_shortest_path_length(graph_dist, username, cutoff=4)
                     if graph_dist.has_node(username) else {})

        u_feats = self._user_feats(username)
        rows = []
        for repo in valid_repos:
            r_feats = self._repo_feats(repo)
            rows.append({
                "username": username, "repo": repo,
                "text_similarity":               self._text_sim(username, repo),
                "language_match":                self._lang_match(username, repo),
                "follows_owner":                 self._follows_owner(username, repo),
                "common_contributors_following": self._common_contrib_following(username, repo),
                "shared_repos_with_owner":       self._shared_repos_owner(username, repo),
                "graph_proximity":               self._graph_prox(distances, repo),
                **r_feats,
                **u_feats,
                "owner_follower_count":          self._owner_follower_count(repo),
                "already_participates": int(repo in self.user_repos.get(username,set())),
            })
        return pd.DataFrame(rows)

    def _is_cold_start(self, username):
        """Retorna True se o usuário não tem conexões no grafo da base."""
        return not self.G.has_node(username) or self.G.degree(username) == 0

    def recommend(self, username, candidate_repos, top_k=10,
                  exclude_existing=True, exclude_archived=True,
                  model=None, weights=None):
        features = self.build_feature_matrix(username, candidate_repos)
        if features.empty:
            return features, False

        cold_start = self._is_cold_start(username)

        if exclude_existing:
            features = features[features["already_participates"] == 0].reset_index(drop=True)
        if exclude_archived:
            features = features[features["repo_is_archived"] == 0].reset_index(drop=True)
        if features.empty:
            print("[info] nenhum candidato restante após filtros.")
            return features, cold_start

        if model is not None and not cold_start:
            # Modelo treinado: usar normalmente
            features["score"] = model.predict_proba(features[FEATURE_COLUMNS].values)[:, 1]
        elif model is not None and cold_start:
            # Cold-start: modelo treinado não tem base social — usar heurística
            # baseada só em features que existem para qualquer repo/usuário
            cold_weights = {
                "text_similarity":    0.30,
                "language_match":     0.25,
                "repo_popularity":    0.15,
                "repo_freshness":     0.15,
                "repo_open_issues":   0.05,
                "user_total_stars":   0.05,
                "owner_follower_count": 0.05,
            }
            score = np.zeros(len(features))
            for feat, w in cold_weights.items():
                if feat not in features.columns:
                    continue
                col = features[feat].values.astype(float)
                normalized = col if feat in ("text_similarity","language_match") else _minmax(col)
                score += w * normalized
            features["score"] = score
        else:
            weights = weights or WEIGHTS
            score = np.zeros(len(features))
            for feat, w in weights.items():
                if feat not in features.columns:
                    continue
                col = features[feat].values.astype(float)
                normalized = col if feat in ("text_similarity","follows_owner",
                                             "language_match","repo_is_fork",
                                             "repo_is_archived","user_pr_ratio",
                                             "user_influence","user_has_readme_profile",
                                             "user_has_company","user_has_blog") else _minmax(col)
                score += w * normalized
            features["score"] = score

        result = features.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)

        # Enriquece com informações legíveis do repositório
        def _repo_info(repo):
            if repo not in self.repos_df.index:
                return "", "", 0, ""
            r = self.repos_df.loc[repo]
            desc    = str(r.get("description") or "")[:60]
            lang    = str(r.get("primary_language") or "")
            stars   = int(_safe(r.get("stars")))
            topics  = str(r.get("topics") or "").replace("|", " ")[:40]
            return desc, lang, stars, topics

        descs, langs, stars_list, topics_list = [], [], [], []
        for repo in result["repo"]:
            d, l, s, t = _repo_info(repo)
            descs.append(d); langs.append(l); stars_list.append(s); topics_list.append(t)

        result.insert(1, "linguagem",   langs)
        result.insert(2, "estrelas",    stars_list)
        result.insert(3, "descricao",   descs)
        result.insert(4, "topicos",     topics_list)

        return result, cold_start


# =========================================================
# DISPLAY
# =========================================================
def _print_result(result, username, top_k, cold_start):
    """Imprime as recomendações de forma legível."""
    BOLD  = "\033[1m"
    CYAN  = "\033[36m"
    GRAY  = "\033[90m"
    RESET = "\033[0m"
    WARN  = "\033[33m"

    print(f"\n{BOLD}Top {top_k} recomendações para '{username}'{RESET}\n")

    if cold_start:
        print(f"{WARN}⚠  Cold-start: '{username}' não tem conexões na base local.")
        print(f"   Features sociais (follows_owner, graph_proximity, etc.) são todas zero.")
        print(f"   O ranking usa texto, linguagem e popularidade do repositório.{RESET}\n")

    if result.empty:
        print("(nenhuma recomendação encontrada)")
        return

    score_std = result["score"].std()
    if score_std < 0.005:
        print(f"{WARN}⚠  Scores muito próximos (desvio padrão = {score_std:.4f}) — "
              f"ranking pode não ser confiável.{RESET}\n")

    # Colunas de features não-zero para mostrar (exclui as zeradas)
    feature_cols = ["text_similarity","language_match","follows_owner",
                    "common_contributors_following","graph_proximity",
                    "repo_freshness","repo_open_issues"]
    nonzero_feats = [c for c in feature_cols
                     if c in result.columns and result[c].abs().max() > 0.001]

    for i, row in result.iterrows():
        rank   = i + 1
        repo   = row["repo"]
        score  = row["score"]
        lang   = row.get("linguagem","") or "?"
        stars  = int(row.get("estrelas", 0))
        desc   = row.get("descricao","") or ""
        topics = row.get("topicos","") or ""

        print(f"{BOLD}#{rank:02d}{RESET}  {CYAN}{repo}{RESET}")
        print(f"     {GRAY}Linguagem:{RESET} {lang:<15} "
              f"{GRAY}⭐ Estrelas:{RESET} {stars:,}")
        if desc:
            print(f"     {GRAY}Descrição:{RESET}  {desc}")
        if topics:
            print(f"     {GRAY}Tópicos:{RESET}    {topics}")

        # Mostra só as features com valor > 0
        feat_parts = []
        for feat in nonzero_feats:
            val = row.get(feat, 0)
            if val > 0.001:
                label = feat.replace("_"," ").replace("repo ","").replace("common ","")
                feat_parts.append(f"{label}: {val:.3f}")
        if feat_parts:
            print(f"     {GRAY}Sinais:  {RESET} {' | '.join(feat_parts)}")

        print(f"     {GRAY}Score:{RESET}     {score:.4f}")
        print()


# =========================================================
# CLI
# =========================================================
def _load_repo_list(args):
    if args.repos:
        return [r.strip() for r in args.repos.split(",") if r.strip()]
    if args.repos_file:
        with open(args.repos_file) as f:
            return [l.strip() for l in f if l.strip()]
    if os.path.exists(REPOS_CSV):
        return pd.read_csv(REPOS_CSV)["full_name"].tolist()
    return []


def main():
    parser = argparse.ArgumentParser(description="Recomenda repositórios GitHub para um usuário.")
    parser.add_argument("--usuario",            required=True)
    parser.add_argument("--repos",              help="lista 'owner/repo' separada por vírgula")
    parser.add_argument("--repos_file",         help="arquivo .txt com um 'owner/repo' por linha")
    parser.add_argument("--top_k",              type=int, default=10)
    parser.add_argument("--modelo",             help="caminho para .joblib gerado por Modeltraining.py")
    parser.add_argument("--incluir_existentes", action="store_true")
    parser.add_argument("--incluir_arquivados", action="store_true")
    args = parser.parse_args()

    rec = Recommender()

    model = None
    if args.modelo:
        bundle = joblib.load(args.modelo)
        model  = bundle["model"]
        saved  = bundle.get("feature_columns",[])
        if saved != FEATURE_COLUMNS:
            print("[aviso] features do modelo diferem das atuais — retreine o modelo.")
        print(f"[info] modelo carregado: {args.modelo}")
    else:
        print("[info] sem modelo — usando score heurístico")

    candidates = _load_repo_list(args)
    if not candidates:
        print("[erro] nenhum repositório candidato. Use --repos ou --repos_file.")
        return

    result, cold_start = rec.recommend(
        args.usuario, candidates,
        top_k            = args.top_k,
        exclude_existing = not args.incluir_existentes,
        exclude_archived = not args.incluir_arquivados,
        model            = model,
    )

    _print_result(result, args.usuario, args.top_k, cold_start)


if __name__ == "__main__":
    main()
