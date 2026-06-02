import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# =========================================================
# CONFIGURAÇÕES
# =========================================================

INPUT_CSV = "./user_repo_dataset_with_text_similarity.csv"

OUTPUT_DIR = "result"

PLOTS_DIR = f"{OUTPUT_DIR}/plots"

REPORTS_DIR = f"{OUTPUT_DIR}/reports"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

RANDOM_SEED = 42

# =========================================================
# CARREGAR DATASET
# =========================================================

print("\nCarregando dataset...")

df = pd.read_csv(INPUT_CSV)

print(f"\nShape: {df.shape}")

# =========================================================
# FEATURES E LABEL
# =========================================================

TARGET_COLUMN = "label"

REMOVE_COLUMNS = ["label", "username", "repo_name"]

X = df.drop(columns=REMOVE_COLUMNS)

y = df[TARGET_COLUMN]

# =========================================================
# TRATAMENTO
# =========================================================

X = X.replace([np.inf, -np.inf], 0)

X = X.fillna(0)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

print("\nSeparando treino/teste...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

print(f"Treino: {X_train.shape}")

print(f"Teste: {X_test.shape}")

# =========================================================
# NORMALIZAÇÃO
# =========================================================

print("\nNormalizando dados...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# =========================================================
# MODELOS
# =========================================================

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=5000, class_weight="balanced", random_state=RANDOM_SEED
    ),
}

# =========================================================
# RESULTADOS
# =========================================================

results = []

# =========================================================
# TREINAMENTO
# =========================================================

for model_name, model in models.items():
    print("\n===================================")

    print(f"Treinando {model_name}")

    print("===================================")

    # =====================================================
    # TREINO
    # =====================================================

    model.fit(X_train_scaled, y_train)

    # =====================================================
    # PREDIÇÕES
    # =====================================================

    y_pred = model.predict(X_test_scaled)

    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # =====================================================
    # MÉTRICAS
    # =====================================================

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    roc_auc = roc_auc_score(y_test, y_prob)

    # =====================================================
    # SALVAR RESULTADOS
    # =====================================================

    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc,
        }
    )

    # =====================================================
    # MATRIZ DE CONFUSÃO
    # =====================================================

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title(f"Confusion Matrix - {model_name}")

    plt.xlabel("Predicted")

    plt.ylabel("True")

    plt.tight_layout()

    plt.savefig(f"{PLOTS_DIR}/{model_name}_confusion_matrix.png")

    plt.close()

    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    report = classification_report(y_test, y_pred)

    with open(f"{REPORTS_DIR}/{model_name}_report.txt", "w") as f:
        f.write(report)

    print("\nClassification Report:\n")

    print(report)

    # =====================================================
    # FEATURE IMPORTANCE RF
    # =====================================================

    if model_name == "RandomForest":
        importances = pd.DataFrame(
            {"feature": X.columns, "importance": model.feature_importances_}
        )

        importances = importances.sort_values(by="importance", ascending=False)

        importances.to_csv(f"{REPORTS_DIR}/rf_feature_importance.csv", index=False)

        # gráfico
        plt.figure(figsize=(10, 8))

        top_features = importances.head(15)

        plt.barh(top_features["feature"], top_features["importance"])

        plt.gca().invert_yaxis()

        plt.title("Top 15 Feature Importances - Random Forest")

        plt.tight_layout()

        plt.savefig(f"{PLOTS_DIR}/rf_feature_importance.png")

        plt.close()

# =========================================================
# DATAFRAME RESULTADOS
# =========================================================

results_df = pd.DataFrame(results)

results_df.to_csv(f"{REPORTS_DIR}/model_comparison.csv", index=False)

print("\n===================================")
print("RESULTADOS")
print("===================================")

print(results_df)

# =========================================================
# COMPARAÇÃO F1
# =========================================================

plt.figure(figsize=(8, 5))

plt.bar(results_df["Model"], results_df["F1"])

plt.title("Comparação F1-Score")

plt.ylabel("F1 Score")

plt.tight_layout()

plt.savefig(f"{PLOTS_DIR}/f1_comparison.png")

plt.close()

# =========================================================
# COMPARAÇÃO ROC AUC
# =========================================================

plt.figure(figsize=(8, 5))

plt.bar(results_df["Model"], results_df["ROC_AUC"])

plt.title("Comparação ROC-AUC")

plt.ylabel("ROC-AUC")

plt.tight_layout()

plt.savefig(f"{PLOTS_DIR}/roc_auc_comparison.png")

plt.close()

# =========================================================
# RESUMO FINAL
# =========================================================

summary_path = f"{REPORTS_DIR}/summary.txt"

with open(summary_path, "w") as f:
    f.write("====================================\n")
    f.write("COMPARAÇÃO DE MODELOS\n")
    f.write("====================================\n\n")

    f.write(results_df.to_string())

print("\n===================================")
print("TREINAMENTO FINALIZADO")
print("===================================")

print(f"\nResultados salvos em:")

print(OUTPUT_DIR)

print("""
Estrutura:

data/models/
├── plots/
│   ├── f1_comparison.png
│   ├── roc_auc_comparison.png
│   ├── RandomForest_confusion_matrix.png
│   ├── LogisticRegression_confusion_matrix.png
│   └── rf_feature_importance.png
│
├── reports/
│   ├── model_comparison.csv
│   ├── RandomForest_report.txt
│   ├── LogisticRegression_report.txt
│   ├── rf_feature_importance.csv
│   └── summary.txt
""")
