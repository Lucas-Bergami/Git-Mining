"""
Análise descritiva da rede de usuários e repositórios do GitHub.

Métricas calculadas:
  - Nós totais e por tipo (usuário / repositório)
  - Arestas totais e por tipo (segue / possui / contribui)
  - Densidade
  - Distribuição de graus (in, out, total) — histograma e top-10
  - Coeficiente de clustering médio
  - Componentes conectados (versão não-dirigida)
  - Nós mais centrais (por grau)

Uso:
    python AnaliseRede.py
    python AnaliseRede.py --data_dir data/raw --salvar_graficos
"""

import os
import argparse
import pandas as pd
import numpy as np
import networkx as nx


DATA_DIR = "data/raw"


# =========================================================
# CARREGAMENTO
# =========================================================
def load_graph(data_dir):
    graphml = f"{data_dir}/graph.graphml"
    if os.path.exists(graphml):
        print(f"[grafo] carregando de {graphml}")
        return nx.read_graphml(graphml)

    # Reconstrói das tabelas de arestas se GraphML não existir
    print("[grafo] GraphML não encontrado — reconstruindo das tabelas CSV...")
    def _read(fname, cols):
        path = f"{data_dir}/{fname}"
        return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=cols)

    users_df   = _read("users_data.csv",      ["username"])
    repos_df   = _read("repos_data.csv",       ["full_name","primary_language","stars"])
    follow_df  = _read("edges_follow.csv",     ["follower","followed"])
    owns_df    = _read("edges_owns.csv",        ["user","repo"])
    contrib_df = _read("edges_contributes.csv", ["user","repo","contributions"])

    G = nx.DiGraph()
    for _, r in users_df.iterrows():
        G.add_node(r["username"], node_type="user")
    for _, r in repos_df.iterrows():
        G.add_node(r["full_name"], node_type="repo",
                   stars=r.get("stars",0), language=r.get("primary_language","Unknown"))
    for _, r in follow_df.iterrows():
        G.add_edge(r["follower"], r["followed"], relation="segue")
    for _, r in owns_df.iterrows():
        G.add_edge(r["user"], r["repo"], relation="possui")
    for _, r in contrib_df.iterrows():
        G.add_edge(r["user"], r["repo"], relation="contribui",
                   contributions=r.get("contributions",0))
    return G


# =========================================================
# ANÁLISE
# =========================================================
def analyze(G):
    print("\n" + "="*60)
    print("  ANÁLISE DESCRITIVA DA REDE")
    print("="*60)

    # --- Nós ---
    n_total  = G.number_of_nodes()
    node_types = nx.get_node_attributes(G, "node_type")
    n_users = sum(1 for t in node_types.values() if t == "user")
    n_repos = sum(1 for t in node_types.values() if t == "repo")
    n_outros = n_total - n_users - n_repos

    print(f"\n{'─'*40}")
    print("  NÓS")
    print(f"{'─'*40}")
    print(f"  Total             : {n_total:,}")
    print(f"  Usuários          : {n_users:,}")
    print(f"  Repositórios      : {n_repos:,}")
    if n_outros:
        print(f"  Outros            : {n_outros:,}")

    # --- Arestas ---
    e_total = G.number_of_edges()
    edge_types = nx.get_edge_attributes(G, "relation")
    n_segue    = sum(1 for r in edge_types.values() if r == "segue")
    n_possui   = sum(1 for r in edge_types.values() if r == "possui")
    n_contrib  = sum(1 for r in edge_types.values() if r == "contribui")
    n_outros_e = e_total - n_segue - n_possui - n_contrib

    print(f"\n{'─'*40}")
    print("  ARESTAS")
    print(f"{'─'*40}")
    print(f"  Total             : {e_total:,}")
    print(f"  segue             : {n_segue:,}")
    print(f"  possui            : {n_possui:,}")
    print(f"  contribui         : {n_contrib:,}")
    if n_outros_e:
        print(f"  outros            : {n_outros_e:,}")

    # --- Densidade ---
    density = nx.density(G)
    print(f"\n{'─'*40}")
    print("  DENSIDADE")
    print(f"{'─'*40}")
    print(f"  Densidade         : {density:.6f}")
    print(f"  (1 = completamente conectado, 0 = sem arestas)")

    # --- Distribuição de graus ---
    in_degrees  = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    degrees     = dict(G.degree())

    in_vals  = list(in_degrees.values())
    out_vals = list(out_degrees.values())
    tot_vals = list(degrees.values())

    print(f"\n{'─'*40}")
    print("  DISTRIBUIÇÃO DE GRAUS")
    print(f"{'─'*40}")
    print(f"  {'':20s}  {'grau total':>10}  {'grau entrada':>12}  {'grau saída':>10}")
    for label, vals in [("Mínimo", [min(tot_vals), min(in_vals), min(out_vals)]),
                         ("Máximo", [max(tot_vals), max(in_vals), max(out_vals)]),
                         ("Média",  [np.mean(tot_vals), np.mean(in_vals), np.mean(out_vals)]),
                         ("Mediana",[np.median(tot_vals), np.median(in_vals), np.median(out_vals)]),
                         ("Desvio padrão", [np.std(tot_vals), np.std(in_vals), np.std(out_vals)])]:
        print(f"  {label:20s}  {vals[0]:>10.2f}  {vals[1]:>12.2f}  {vals[2]:>10.2f}")

    # Histograma ASCII de distribuição de grau total
    print(f"\n  Histograma de grau total (nós em log-escala):")
    _ascii_hist(tot_vals)

    # --- Top 10 por grau total ---
    top10 = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n{'─'*40}")
    print("  TOP 10 NÓS (por grau total)")
    print(f"{'─'*40}")
    print(f"  {'nó':<40} {'tipo':<10} {'grau':>6} {'entrada':>8} {'saída':>6}")
    for node, deg in top10:
        ntype = node_types.get(node, "?")
        label = node if len(node) <= 38 else node[:35]+"..."
        print(f"  {label:<40} {ntype:<10} {deg:>6}  {in_degrees[node]:>7}  {out_degrees[node]:>5}")

    # --- Coeficiente de clustering ---
    G_und = G.to_undirected()
    # Amostra se grafo for grande (clustering é O(n*k²))
    if n_total > 5000:
        sample_nodes = list(G_und.nodes)[:5000]
        G_sample = G_und.subgraph(sample_nodes)
        cc = nx.average_clustering(G_sample)
        note = " (amostra de 5000 nós)"
    else:
        cc = nx.average_clustering(G_und)
        note = ""
    print(f"\n{'─'*40}")
    print("  COEFICIENTE DE CLUSTERING")
    print(f"{'─'*40}")
    print(f"  Coef. clustering médio : {cc:.4f}{note}")
    print(f"  (0 = nenhum triângulo, 1 = todos vizinhos conectados entre si)")

    # --- Sub-grafo apenas de usuários (relação 'segue') ---
    user_nodes = [n for n, t in node_types.items() if t == "user"]
    G_users = G.subgraph(user_nodes)
    cc_users = nx.average_clustering(G_users.to_undirected()) if G_users.number_of_nodes() > 0 else 0.0
    print(f"  Coef. clustering (só usuários, só 'segue'): {cc_users:.4f}")

    # --- Componentes conectados ---
    components = list(nx.connected_components(G_und))
    n_comp = len(components)
    largest = max(len(c) for c in components) if components else 0
    print(f"\n{'─'*40}")
    print("  COMPONENTES CONECTADOS (grafo não-dirigido)")
    print(f"{'─'*40}")
    print(f"  Número de componentes          : {n_comp:,}")
    print(f"  Maior componente               : {largest:,} nós "
          f"({100*largest/n_total:.1f}% do total)")
    print(f"  Componentes de tamanho 1 (isolados): "
          f"{sum(1 for c in components if len(c)==1):,}")

    print(f"\n{'='*60}\n")
    return G


def _ascii_hist(values, n_bins=10, width=40):
    """Histograma ASCII simples."""
    arr = np.array(values)
    counts, edges = np.histogram(arr, bins=n_bins)
    max_count = max(counts) if max(counts) > 0 else 1
    for i, (cnt, edge) in enumerate(zip(counts, edges)):
        bar_len = int(width * cnt / max_count)
        label   = f"{int(edge):>5}-{int(edges[i+1]):<5}"
        print(f"  {label} | {'█'*bar_len} {cnt}")


# =========================================================
# PLOTS (opcional, requer matplotlib)
# =========================================================
def save_plots(G, out_dir="data/analise"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[aviso] matplotlib não instalado — gráficos não gerados. pip install matplotlib")
        return

    os.makedirs(out_dir, exist_ok=True)

    degrees  = [d for _, d in G.degree()]
    in_degs  = [d for _, d in G.in_degree()]
    out_degs = [d for _, d in G.out_degree()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Distribuição de Graus da Rede GitHub", fontsize=13)

    for ax, data, title in zip(axes,
                                [degrees, in_degs, out_degs],
                                ["Grau Total", "Grau de Entrada", "Grau de Saída"]):
        ax.hist(data, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel("Grau")
        ax.set_ylabel("Frequência")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = f"{out_dir}/distribuicao_graus.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot] salvo em {path}")

    # Distribuição log-log (lei de potência)
    deg_vals = np.array(degrees)
    unique, counts = np.unique(deg_vals[deg_vals > 0], return_counts=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(unique, counts, s=10, alpha=0.6, color="#4C72B0")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Grau (log)"); ax.set_ylabel("Frequência (log)")
    ax.set_title("Distribuição de Graus — escala log-log")
    ax.grid(alpha=0.3)
    path2 = f"{out_dir}/loglog_graus.png"
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"[plot] salvo em {path2}")


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Análise descritiva da rede GitHub coletada.")
    parser.add_argument("--data_dir",        default=DATA_DIR)
    parser.add_argument("--salvar_graficos", action="store_true",
                         help="salva gráficos em data/analise/ (requer matplotlib)")
    args = parser.parse_args()

    G = load_graph(args.data_dir)
    analyze(G)

    if args.salvar_graficos:
        save_plots(G)


if __name__ == "__main__":
    main()
