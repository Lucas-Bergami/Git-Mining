"""
Recomendador: dado um usuário A e uma lista de repositórios candidatos,
devolve os top_K repositórios mais recomendados.

Como ainda não há um modelo treinado, o score é uma combinação heurística
(ponderada) de features tabulares, textuais (TF-IDF) e de grafo. A função
`build_feature_matrix()` devolve as features "cruas" por par (usuário, repo)
— essa matriz pode ser reaproveitada depois para treinar/alimentar um
classificador real, bastando substituir a combinação heurística por
`model.predict_proba(X)`.

Observação sobre o grafo: aqui ele é remontado a partir das tabelas de
arestas (edges_follow / edges_owns / edges_contributes) usando um
MultiDiGraph, em vez do DiGraph simples do script de coleta. Isso evita
perder informação quando o mesmo par (usuário, repo) tem mais de uma
relação — por exemplo, alguém que é dono E aparece como contribuidor do
próprio repositório, caso bem comum.

Features usadas:
  - text_similarity:               cosseno entre TF-IDF do README de perfil do usuário
                                    e do README do repositório
  - language_match:                Jaccard entre linguagens do usuário e do repositório
  - follows_owner:                 usuário segue o dono do repositório?
  - common_contributors_following: quantos contribuidores do repo o usuário já segue
  - shared_repos_with_owner:       em quantos outros repos usuário e dono já
                                    apareceram juntos (contribuição/posse)
  - graph_proximity:               1 / (1 + distância) entre usuário e repo no grafo
  - repo_popularity:               log(estrelas + forks + watchers) do repositório

Uso via linha de comando:
    python recomendar.py --usuario torvalds --top_k 10
    python recomendar.py --usuario torvalds --repos "owner/repo1,owner/repo2"
    python recomendar.py --usuario torvalds --repos_file candidatos.txt

Requisitos: pip install pandas numpy networkx scikit-learn
"""

import os
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "data/raw"
USERS_CSV = f"{DATA_DIR}/users.csv"
REPOS_CSV = f"{DATA_DIR}/repos.csv"

# Pesos do score heurístico — ajuste à mão ou, melhor ainda, substitua o
# método `recommend()` por um modelo treinado usando build_feature_matrix().
WEIGHTS = {
    "text_similarity": 0.25,
    "language_match": 0.15,
    "follows_owner": 0.10,
    "common_contributors_following": 0.15,
    "shared_repos_with_owner": 0.10,
    "graph_proximity": 0.15,
    "repo_popularity": 0.10,
}

# Mesma ordem de colunas usada para treinar (treinar_modelo.py) e para servir o modelo aqui.
FEATURE_COLUMNS = list(WEIGHTS.keys())


def _minmax(values):
    values = np.asarray(values, dtype=float)
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-9:
        return np.full_like(values, 0.5)
    return (values - lo) / (hi - lo)


def _split_set(text):
    if not isinstance(text, str) or not text:
        return set()
    return set(text.split("|"))


def _jaccard(a, b):
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class Recommender:
    def __init__(self, data_dir=DATA_DIR):
        self.users_df = pd.read_csv(f"{data_dir}/users.csv")
        self.repos_df = pd.read_csv(f"{data_dir}/repos.csv")
        self.follow_df = self._read_or_empty(
            f"{data_dir}/edges_follow.csv", ["follower", "followed"]
        )
        self.owns_df = self._read_or_empty(
            f"{data_dir}/edges_owns.csv", ["user", "repo"]
        )
        self.contrib_df = self._read_or_empty(
            f"{data_dir}/edges_contributes.csv", ["user", "repo", "contributions"]
        )

        self.users_df = self.users_df.drop_duplicates("username").set_index(
            "username", drop=False
        )
        self.repos_df = self.repos_df.drop_duplicates("full_name").set_index(
            "full_name", drop=False
        )

        self._build_graph()
        self._build_lookup_sets()
        self._fit_tfidf()

    @staticmethod
    def _read_or_empty(path, cols):
        return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=cols)

    # ---------- estruturas auxiliares ----------
    def _build_graph(self):
        G = nx.MultiDiGraph()
        for username in self.users_df["username"]:
            G.add_node(username, node_type="user")
        for full_name in self.repos_df["full_name"]:
            G.add_node(full_name, node_type="repo")
        for _, row in self.follow_df.iterrows():
            G.add_edge(row["follower"], row["followed"], relation="segue")
        for _, row in self.owns_df.iterrows():
            G.add_edge(row["user"], row["repo"], relation="possui")
        for _, row in self.contrib_df.iterrows():
            G.add_edge(row["user"], row["repo"], relation="contribui")
        self.G = G
        self.G_undirected = G.to_undirected()

    def _build_lookup_sets(self):
        self.following = (
            self.follow_df.groupby("follower")["followed"].apply(set).to_dict()
            if not self.follow_df.empty
            else {}
        )
        self.contributors_of = (
            self.contrib_df.groupby("repo")["user"].apply(set).to_dict()
            if not self.contrib_df.empty
            else {}
        )
        contrib_repos = (
            self.contrib_df.groupby("user")["repo"].apply(set).to_dict()
            if not self.contrib_df.empty
            else {}
        )
        owns_repos = (
            self.owns_df.groupby("user")["repo"].apply(set).to_dict()
            if not self.owns_df.empty
            else {}
        )
        self.user_repos = {}
        for u in set(contrib_repos) | set(owns_repos):
            self.user_repos[u] = contrib_repos.get(u, set()) | owns_repos.get(u, set())

    def _fit_tfidf(self):
        user_texts = (
            self.users_df.get("profile_readme_text", pd.Series(dtype=str))
            .fillna("")
            .tolist()
        )
        repo_texts = (
            self.repos_df.get("readme_text", pd.Series(dtype=str)).fillna("").tolist()
        )
        corpus = user_texts + repo_texts
        if not any(t.strip() for t in corpus):
            corpus = ["sem conteudo"]
        self.vectorizer = TfidfVectorizer(max_features=20000)
        self.vectorizer.fit(corpus)
        self.user_vectors = (
            self.vectorizer.transform(user_texts) if user_texts else None
        )
        self.repo_vectors = (
            self.vectorizer.transform(repo_texts) if repo_texts else None
        )
        self._user_row_index = {u: i for i, u in enumerate(self.users_df["username"])}
        self._repo_row_index = {r: i for i, r in enumerate(self.repos_df["full_name"])}

    # ---------- features por par (usuário, repositório) ----------
    def _text_similarity(self, username, repo):
        ui = self._user_row_index.get(username)
        ri = self._repo_row_index.get(repo)
        if ui is None or ri is None or self.user_vectors is None:
            return 0.0
        sim = cosine_similarity(self.user_vectors[ui], self.repo_vectors[ri])
        return float(sim[0, 0])

    def _language_match(self, username, repo):
        user_langs = (
            _split_set(self.users_df.at[username, "languages"])
            if username in self.users_df.index
            else set()
        )
        repo_langs = (
            _split_set(self.repos_df.at[repo, "languages"])
            if repo in self.repos_df.index
            else set()
        )
        return _jaccard(user_langs, repo_langs)

    def _follows_owner(self, username, repo):
        if repo not in self.repos_df.index:
            return 0.0
        owner = self.repos_df.at[repo, "owner"]
        return 1.0 if owner in self.following.get(username, set()) else 0.0

    def _common_contributors_following(self, username, repo):
        contributors = self.contributors_of.get(repo, set())
        followed = self.following.get(username, set())
        return len(contributors & followed)

    def _shared_repos_with_owner(self, username, repo):
        if repo not in self.repos_df.index:
            return 0
        owner = self.repos_df.at[repo, "owner"]
        if owner == username:
            return 0
        user_repos = self.user_repos.get(username, set())
        owner_repos = self.user_repos.get(owner, set())
        return len((user_repos & owner_repos) - {repo})

    @staticmethod
    def _graph_proximity(distances, repo):
        d = distances.get(repo)
        return 0.0 if d is None else 1.0 / (1.0 + d)

    def _repo_popularity(self, repo):
        if repo not in self.repos_df.index:
            return 0.0
        row = self.repos_df.loc[repo]
        return float(
            np.log1p(row.get("stars", 0) + row.get("forks", 0) + row.get("watchers", 0))
        )

    # ---------- API pública ----------
    def build_feature_matrix(self, username, candidate_repos, unlink_repos=None):
        """Features cruas (não normalizadas) por par (usuário, repo).
        Reaproveitável para treinar/alimentar um modelo de ML no futuro.

        `unlink_repos`: conjunto opcional de repositórios para os quais a
        aresta 'contribui' (usuário -> repo) deve ser removida do grafo antes
        de calcular graph_proximity. Usado durante a montagem da tabela de
        treino, para não deixar a própria aresta que está sendo rotulada
        vazar como feature (veja treinar_modelo.py)."""
        if username not in self.users_df.index:
            raise ValueError(f"Usuário '{username}' não encontrado em {USERS_CSV}.")

        valid_repos = [r for r in candidate_repos if r in self.repos_df.index]
        missing = sorted(set(candidate_repos) - set(valid_repos))
        if missing:
            preview = missing[:5]
            print(
                f"[aviso] {len(missing)} repositório(s) não encontrados na base, ignorados: {preview}"
                f"{' ...' if len(missing) > 5 else ''}"
            )

        graph_for_distance = self.G_undirected
        if unlink_repos:
            graph_for_distance = self.G_undirected.copy()
            for repo in unlink_repos:
                edge_data = graph_for_distance.get_edge_data(username, repo)
                if not edge_data:
                    continue
                keys_to_remove = [
                    k
                    for k, attrs in edge_data.items()
                    if attrs.get("relation") == "contribui"
                ]
                for k in keys_to_remove:
                    graph_for_distance.remove_edge(username, repo, k)

        distances = (
            nx.single_source_shortest_path_length(
                graph_for_distance, username, cutoff=4
            )
            if graph_for_distance.has_node(username)
            else {}
        )

        rows = []
        for repo in valid_repos:
            rows.append(
                {
                    "username": username,
                    "repo": repo,
                    "text_similarity": self._text_similarity(username, repo),
                    "language_match": self._language_match(username, repo),
                    "follows_owner": self._follows_owner(username, repo),
                    "common_contributors_following": self._common_contributors_following(
                        username, repo
                    ),
                    "shared_repos_with_owner": self._shared_repos_with_owner(
                        username, repo
                    ),
                    "graph_proximity": self._graph_proximity(distances, repo),
                    "repo_popularity": self._repo_popularity(repo),
                    "already_participates": int(
                        repo in self.user_repos.get(username, set())
                    ),
                }
            )
        return pd.DataFrame(rows)

    def recommend(
        self,
        username,
        candidate_repos,
        top_k=10,
        exclude_existing=True,
        weights=None,
        model=None,
    ):
        features = self.build_feature_matrix(username, candidate_repos)
        if features.empty:
            return features

        if exclude_existing:
            features = features[features["already_participates"] == 0].reset_index(
                drop=True
            )
        if features.empty:
            print(
                "[info] nenhum candidato restante após remover repositórios em que o usuário já participa."
            )
            return features

        if model is not None:
            X = features[FEATURE_COLUMNS].values
            features["score"] = model.predict_proba(X)[:, 1]
        else:
            weights = weights or WEIGHTS
            score = np.zeros(len(features))
            for feat, w in weights.items():
                if feat in ("text_similarity", "follows_owner"):
                    normalized = features[feat].values  # já estão em [0, 1]
                else:
                    normalized = _minmax(features[feat].values)
                features[f"{feat}_norm"] = normalized
                score += w * normalized
            features["score"] = score

        return (
            features.sort_values("score", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )


# =========================================================
# CLI
# =========================================================
def _load_repo_list(args):
    if args.repos:
        return [r.strip() for r in args.repos.split(",") if r.strip()]
    if args.repos_file:
        with open(args.repos_file) as f:
            return [line.strip() for line in f if line.strip()]
    repos_df = pd.read_csv(REPOS_CSV)
    return repos_df["full_name"].tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Recomenda repositórios do GitHub para um usuário."
    )
    parser.add_argument(
        "--usuario", required=True, help="username do GitHub (deve estar em users.csv)"
    )
    parser.add_argument("--repos", help="lista 'owner/repo' separada por vírgula")
    parser.add_argument(
        "--repos_file", help="arquivo .txt com um 'owner/repo' por linha"
    )
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--incluir_existentes",
        action="store_true",
        help="não remove repositórios em que o usuário já participa",
    )
    parser.add_argument(
        "--modelo",
        help="caminho para um modelo treinado (.joblib, gerado por treinar_modelo.py)",
    )
    args = parser.parse_args()

    rec = Recommender()

    model = None
    if args.modelo:
        bundle = joblib.load(args.modelo)
        model = bundle["model"]
        if bundle.get("feature_columns") != FEATURE_COLUMNS:
            print(
                "[aviso] a ordem das features do modelo salvo é diferente da atual do script — "
                "os resultados podem ficar incorretos."
            )
        print(f"[info] usando modelo treinado: {args.modelo}")
    else:
        print(
            "[info] nenhum modelo informado, usando score heurístico (TF-IDF + grafo)"
        )

    candidates = _load_repo_list(args)
    result = rec.recommend(
        args.usuario,
        candidates,
        top_k=args.top_k,
        exclude_existing=not args.incluir_existentes,
        model=model,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    cols = [
        "repo",
        "score",
        "text_similarity",
        "language_match",
        "follows_owner",
        "common_contributors_following",
        "shared_repos_with_owner",
        "graph_proximity",
        "repo_popularity",
    ]
    print(f"\nTop {args.top_k} recomendações para '{args.usuario}':\n")
    if result.empty:
        print("(nenhuma recomendação encontrada)")
    else:
        print(result[cols].to_string(index=False))


if __name__ == "__main__":
    main()
