import os
import time
import base64
import requests
import pandas as pd

from tqdm import tqdm
from dotenv import load_dotenv

# ==================================
# CONFIGURAÇÕES
# ==================================

INPUT_CSV = "./data/raw/repos.csv"
OUTPUT_CSV = "repos_textos.csv"

LIMIT = None

# ==================================
# TOKEN
# ==================================

load_dotenv()

TOKEN = os.getenv("GITHUB_TOKEN")

if not TOKEN:
    raise ValueError("GITHUB_TOKEN não encontrado no arquivo .env")

headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {TOKEN}"}

# ==================================
# FUNÇÕES
# ==================================


def get_repo_description(owner: str, repo: str) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}"

    try:
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code != 200:
            return ""

        data = response.json()

        return data.get("description", "") or ""

    except Exception as e:
        print(f"Erro descrição {owner}/{repo}: {e}")
        return ""


def get_repo_readme(owner: str, repo: str) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"

    try:
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code != 200:
            return ""

        data = response.json()

        content = data.get("content", "")

        if not content:
            return ""

        decoded = base64.b64decode(content)

        return decoded.decode("utf-8", errors="ignore")

    except Exception as e:
        print(f"Erro README {owner}/{repo}: {e}")
        return ""


# ==================================
# LEITURA DOS REPOSITÓRIOS
# ==================================

df = pd.read_csv(INPUT_CSV)

if LIMIT is not None:
    df = df.head(LIMIT)

# ==================================
# COLETA
# ==================================

results = []

for repo_full in tqdm(df["repo_name"], desc="Coletando repositórios"):
    try:
        owner, repo = repo_full.split("/", 1)

        description = get_repo_description(owner, repo)

        readme = get_repo_readme(owner, repo)

        full_text = f"{description}\n\n{readme}".strip()

        results.append(
            {
                "repo_name": repo_full,
                "description": description,
                "readme": readme,
                "full_text": full_text,
            }
        )

        time.sleep(0.1)

    except Exception as e:
        print(f"Erro em {repo_full}: {e}")

# ==================================
# SALVAR
# ==================================

output_df = pd.DataFrame(results)

output_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(f"\nArquivo salvo em: {OUTPUT_CSV}")
print(f"Repositórios processados: {len(output_df)}")
