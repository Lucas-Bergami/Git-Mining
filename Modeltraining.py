"""
Treina um classificador para prever se um usuário contribui em um repositório,
usando exatamente as mesmas features que o recomendar.py calcula
(Recommender.build_feature_matrix), para garantir consistência entre
treino e inferência.

Rótulos:
  - positivo (1): pares (usuário, repo) presentes em edges_contributes.csv
  - negativo (0): repositórios sorteados, em duas categorias:
      * "difíceis": repositórios de pessoas que o usuário segue mas em que
        nunca contribuiu — testam se o modelo aprende mais do que só
        "esse repo é de alguém famoso"
      * "fáceis": repositórios sorteados aleatoriamente do restante da base

Divisão treino/teste por usuário (GroupShuffleSplit): nenhum usuário aparece
nos dois conjuntos ao mesmo tempo, para evitar que o modelo "memorize"
características de um usuário específico em vez de aprender o padrão geral.

Modelo: XGBoost (gradient boosting em árvores), com regressão logística como
baseline de comparação. Se o xgboost não estiver instalado, cai para
RandomForest automaticamente.

Uso:
    python treinar_modelo.py --neg_por_positivo 4 --saida data/model/contrib_model.joblib
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)

from Recomendar import Recommender, FEATURE_COLUMNS

RANDOM_STATE = 42


def sample_negatives(rec, username, exclude, n_random, n_hard, rng):
    """Sorteia repositórios negativos para um usuário, priorizando 'difíceis' primeiro."""
    pool = set(rec.repos_df["full_name"]) - exclude
    if not pool:
        return []

    followed = rec.following.get(username, set())
    hard_pool = [r for r in pool if rec.repos_df.at[r, "owner"] in followed]
    hard_sample = (
        list(rng.choice(hard_pool, size=min(n_hard, len(hard_pool)), replace=False))
        if hard_pool
        else []
    )

    remaining_pool = list(pool - set(hard_sample))
    random_sample = (
        list(
            rng.choice(
                remaining_pool, size=min(n_random, len(remaining_pool)), replace=False
            )
        )
        if remaining_pool and n_random > 0
        else []
    )

    return hard_sample + random_sample


def build_training_table(rec, neg_por_positivo=4, hard_frac=0.5):
    rng = np.random.default_rng(RANDOM_STATE)

    contrib_only = rec.contrib_df.groupby("user")["repo"].apply(set).to_dict()
    owns_only = rec.owns_df.groupby("user")["repo"].apply(set).to_dict()
    users_with_contrib = sorted(set(contrib_only) & set(rec.users_df.index))
    print(
        f"Montando tabela de treino para {len(users_with_contrib)} usuários com pelo menos 1 contribuição..."
    )

    skipped_self_only = 0
    rows = []
    for username in users_with_contrib:
        # Só interessa contribuição em repositório de terceiros — contribuir no
        # próprio repo é trivial (o GitHub já lista o dono como contribuidor) e
        # criaria vazamento via a aresta "possui".
        positives = sorted(contrib_only[username] - owns_only.get(username, set()))
        if not positives:
            skipped_self_only += 1
            continue

        n_neg = max(1, len(positives) * neg_por_positivo)
        n_hard = int(round(n_neg * hard_frac))
        n_random = n_neg - n_hard

        exclude = rec.user_repos.get(username, set()) | set(positives)
        negatives = sample_negatives(rec, username, exclude, n_random, n_hard, rng)
        if not negatives:
            continue

        label_map = {r: 1 for r in positives}
        label_map.update({r: 0 for r in negatives})

        # unlink_repos remove a aresta "contribui" dos positivos antes de calcular
        # graph_proximity — sem isso, o grafo conteria a própria aresta que estamos
        # tentando prever (vazamento direto do rótulo na feature).
        feats = rec.build_feature_matrix(
            username, positives + negatives, unlink_repos=set(positives)
        )
        if feats.empty:
            continue
        feats["label"] = feats["repo"].map(label_map)
        rows.append(feats)

    if skipped_self_only:
        print(
            f"[info] {skipped_self_only} usuário(s) só tinham contribuição no próprio repositório "
            f"e foram ignorados (não geram exemplo de terceiros)."
        )

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description="Treina o modelo de previsão de contribuição usuário-repositório."
    )
    parser.add_argument(
        "--neg_por_positivo",
        type=int,
        default=4,
        help="quantos negativos sortear para cada exemplo positivo",
    )
    parser.add_argument(
        "--hard_frac",
        type=float,
        default=0.5,
        help="fração dos negativos que são 'difíceis' (repos de quem o usuário segue)",
    )
    parser.add_argument("--saida", default="data/model/contrib_model.joblib")
    args = parser.parse_args()

    rec = Recommender()
    table = build_training_table(rec, args.neg_por_positivo, args.hard_frac)
    if table.empty:
        print(
            "Não foi possível montar exemplos de treino (nenhum usuário com contribuições registradas?)."
        )
        return

    n_pos = int(table["label"].sum())
    n_neg = int((table["label"] == 0).sum())
    print(
        f"\nTabela de treino: {len(table)} exemplos | positivos: {n_pos} | negativos: {n_neg}"
    )

    X = table[FEATURE_COLUMNS].values
    y = table["label"].values
    groups = table["username"].values

    if len(set(groups)) < 4:
        print(
            "[aviso] poucos usuários distintos para um split por grupo confiável; "
            "os resultados de avaliação abaixo servem só de referência, não como avaliação final."
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(
        f"Split por usuário (nenhum usuário em comum entre treino e teste): "
        f"{len(train_idx)} treino, {len(test_idx)} teste"
    )

    try:
        from xgboost import XGBClassifier

        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
        model_name = "XGBoost"
    except ImportError:
        print(
            "[aviso] xgboost não instalado (pip install xgboost). Usando RandomForest no lugar."
        )
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        model_name = "RandomForest"

    model.fit(X_train, y_train)

    baseline = LogisticRegression(max_iter=1000, class_weight="balanced")
    baseline.fit(X_train, y_train)

    for name, m in [(model_name, model), ("Regressão logística (baseline)", baseline)]:
        proba = m.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        print(f"\n=== {name} ===")
        print(classification_report(y_test, pred, digits=3, zero_division=0))
        print(
            f"ROC-AUC: {roc_auc_score(y_test, proba):.3f} | PR-AUC: {average_precision_score(y_test, proba):.3f}"
        )

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(
            model.feature_importances_, index=FEATURE_COLUMNS
        ).sort_values(ascending=False)
        print("\nImportância das features:")
        print(importances.to_string())

    os.makedirs(os.path.dirname(args.saida) or ".", exist_ok=True)
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, args.saida)
    print(f"\nModelo salvo em {args.saida}")
    print(
        f"Para usar no recomendador: python recomendar.py --usuario <user> --modelo {args.saida}"
    )


if __name__ == "__main__":
    main()
