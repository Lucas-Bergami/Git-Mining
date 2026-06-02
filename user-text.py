import os
import re
import sys
import time
import base64
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# =========================================================
# USO:
#   python fetch_texts.py                  → pega todos os usuários
#   python fetch_texts.py 50               → pega os primeiros 50
#   python fetch_texts.py 100 data/raw/users.csv  → CSV customizado
# =========================================================

load_dotenv()

TOKEN   = os.getenv("GITHUB_TOKEN")
headers = {"Authorization": f"token {TOKEN}"}

# =========================================================
# PARÂMETROS
# =========================================================

LIMIT    = int(sys.argv[1]) if len(sys.argv) > 1 else None
CSV_PATH = sys.argv[2] if len(sys.argv) > 2 else "data/raw/users.csv"
OUT_PATH = "data/raw/users_texts.csv"

os.makedirs("data/raw", exist_ok=True)

# =========================================================
# RATE LIMIT
# =========================================================

def check_rate_limit():
    r = requests.get("https://api.github.com/rate_limit", headers=headers)
    if r.status_code == 200:
        d = r.json()
        return d["rate"]["remaining"], d["rate"]["reset"]
    return None, None


class RateLimitException(Exception):
    pass


def github_get(url):
    response = requests.get(url, headers=headers)

    if response.status_code in (403, 429):
        remaining, reset_at = check_rate_limit()
        reset_time = datetime.fromtimestamp(reset_at).strftime("%H:%M:%S") if reset_at else "?"
        print(f"\n[RATE LIMIT] Restantes: {remaining}. Reset às {reset_time}.")
        raise RateLimitException("Rate limit atingido.")

    if response.status_code == 200:
        time.sleep(0.3)
        return response.json()

    return None


def get_profile_readme(username):
    """
    Busca o README do repositório de perfil (username/username).
    Retorna o texto bruto decodificado ou None se não existir.
    """
    url      = f"https://api.github.com/repos/{username}/{username}/readme"
    response = requests.get(url, headers=headers)

    if response.status_code in (403, 429):
        remaining, reset_at = check_rate_limit()
        reset_time = datetime.fromtimestamp(reset_at).strftime("%H:%M:%S") if reset_at else "?"
        print(f"\n[RATE LIMIT] Restantes: {remaining}. Reset às {reset_time}.")
        raise RateLimitException("Rate limit atingido.")

    if response.status_code == 200:
        time.sleep(0.3)
        data     = response.json()
        content  = data.get("content", "")
        encoding = data.get("encoding", "")
        if encoding == "base64":
            return base64.b64decode(content).decode("utf-8", errors="ignore")
        return content

    return None

# =========================================================
# LIMPEZA DE TEXTO PARA TF-IDF
# =========================================================

STOP_WORDS = set(stopwords.words("english"))
stemmer    = PorterStemmer()

def clean_text(text):
    """
    Limpa e normaliza o texto para uso com TF-IDF:
      1. Lowercase
      2. Remove blocos de código markdown
      3. Remove headers markdown
      4. Remove URLs
      5. Remove emojis e caracteres não-ASCII
      6. Remove menções (@user) e hashtags (#tag)
      7. Remove pontuação e números
      8. Tokeniza, remove stopwords e tokens curtos (<=2 chars)
      9. Stemming
    Retorna None se o texto resultante estiver vazio.
    """
    if not text or not isinstance(text, str):
        return None

    text = text.lower()
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"^#{1,6}\s+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens]

    result = " ".join(tokens).strip()
    return result if result else None

# =========================================================
# LEITURA DO CSV DE USUÁRIOS
# =========================================================

print(f"\nLendo: {CSV_PATH}")
users_df  = pd.read_csv(CSV_PATH)
usernames = users_df["username"].dropna().unique().tolist()

if LIMIT is not None:
    usernames = usernames[:LIMIT]

print(f"Usuários a processar: {len(usernames)}")

# =========================================================
# RETOMADA
# =========================================================

if os.path.exists(OUT_PATH):
    existing_df  = pd.read_csv(OUT_PATH)
    already_done = set(existing_df["username"].tolist())
    texts_data   = existing_df.to_dict(orient="records")
    print(f"Retomando: {len(already_done)} já coletados.")
else:
    already_done = set()
    texts_data   = []

# =========================================================
# COLETA E LIMPEZA
# =========================================================

print("\nColetando bio e README de perfil...\n")

try:
    for i, username in enumerate(usernames, 1):
        if username in already_done:
            print(f"  [{i}/{len(usernames)}] {username} — já coletado, pulando.")
            continue

        print(f"  [{i}/{len(usernames)}] {username}", end=" ... ")

        # Bio via API de usuário
        user        = github_get(f"https://api.github.com/users/{username}")
        raw_bio     = user.get("bio") if user else None
        cleaned_bio = clean_text(raw_bio)

        # README de perfil
        raw_readme     = get_profile_readme(username)
        cleaned_readme = clean_text(raw_readme)

        # Texto combinado: bio + readme (concatenados com espaço)
        parts        = [t for t in [cleaned_bio, cleaned_readme] if t]
        combined     = " ".join(parts) if parts else None

        texts_data.append({
            "username": username,
            "bio":      cleaned_bio,     # bio isolada (pode ser None)
            "readme":   cleaned_readme,  # readme isolado (pode ser None)
            "text":     combined,        # bio + readme combinados para TF-IDF
        })

        bio_s    = f"{len(cleaned_bio.split())} tokens"    if cleaned_bio    else "sem bio"
        readme_s = f"{len(cleaned_readme.split())} tokens" if cleaned_readme else "sem readme"
        print(f"bio: {bio_s} | readme: {readme_s}")

        if len(texts_data) % 10 == 0:
            pd.DataFrame(texts_data).to_csv(OUT_PATH, index=False)
            print(f"  [checkpoint] {OUT_PATH}")

except RateLimitException:
    print("\n[RATE LIMIT] Salvando progresso...")
finally:
    pd.DataFrame(texts_data).to_csv(OUT_PATH, index=False)

# =========================================================
# RESULTADO
# =========================================================

result_df  = pd.read_csv(OUT_PATH)
total      = len(result_df)
com_bio    = result_df["bio"].notna().sum()
com_readme = result_df["readme"].notna().sum()
com_texto  = result_df["text"].notna().sum()

print(f"\nArquivo salvo         : {OUT_PATH}")
print(f"Total processados     : {total}")
print(f"Com bio               : {com_bio}")
print(f"Com README            : {com_readme}")
print(f"Com algum texto       : {com_texto}")
print(f"Sem nenhum texto      : {total - com_texto}")

if com_texto > 0:
    print(f"\nExemplos:")
    print(result_df[result_df["text"].notna()][["username", "bio", "readme"]].head(5).to_string(index=False))