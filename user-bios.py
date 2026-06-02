import os
import re
import sys
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Baixa recursos do NLTK se necessário
nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# =========================================================
# USO:
#   python fetch_bios.py                  → pega todos os usuários
#   python fetch_bios.py 50               → pega os primeiros 50
#   python fetch_bios.py 100 data/raw/users.csv  → CSV customizado
# =========================================================

load_dotenv()

TOKEN   = os.getenv("GITHUB_TOKEN")
headers = {"Authorization": f"token {TOKEN}"}

# =========================================================
# PARÂMETROS VIA LINHA DE COMANDO
# =========================================================

LIMIT    = None
CSV_PATH = sys.argv[2] if len(sys.argv) > 2 else "data/raw/users.csv"
OUT_PATH = "data/raw/users_bios.csv"

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

# =========================================================
# LIMPEZA DE TEXTO PARA TF-IDF
# =========================================================

STOP_WORDS = set(stopwords.words("english"))
stemmer    = PorterStemmer()

def clean_bio(text):
    """
    Limpa e normaliza o texto da bio para uso com TF-IDF:
      1. Lowercase
      2. Remove URLs
      3. Remove emojis e caracteres não-ASCII
      4. Remove menções (@user) e hashtags (#tag)
      5. Remove pontuação e números
      6. Tokeniza e remove stopwords (palavras com <= 2 chars também)
      7. Stemming (reduz palavras à raiz)
    Retorna None se o texto resultante estiver vazio.
    """
    if not text or not isinstance(text, str):
        return None

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 3. Remove emojis e caracteres não-ASCII
    text = text.encode("ascii", "ignore").decode("ascii")

    # 4. Remove menções e hashtags
    text = re.sub(r"@\w+|#\w+", "", text)

    # 5. Remove pontuação e números
    text = re.sub(r"[^a-z\s]", "", text)

    # 6. Tokeniza, remove stopwords e tokens muito curtos
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    # 7. Stemming
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
# RETOMADA — carrega output existente
# =========================================================

if os.path.exists(OUT_PATH):
    existing_df  = pd.read_csv(OUT_PATH)
    already_done = set(existing_df["username"].tolist())
    bios_data    = existing_df.to_dict(orient="records")
    print(f"Retomando: {len(already_done)} bios já coletadas.")
else:
    already_done = set()
    bios_data    = []

# =========================================================
# COLETA E LIMPEZA
# =========================================================

print("\nColetando e limpando bios...\n")

try:
    for i, username in enumerate(usernames, 1):
        if username in already_done:
            print(f"  [{i}/{len(usernames)}] {username} — já coletado, pulando.")
            continue

        print(f"  [{i}/{len(usernames)}] {username}", end=" ... ")

        user = github_get(f"https://api.github.com/users/{username}")

        if user is None:
            print("erro na requisição, pulando.")
            continue

        raw_bio     = user.get("bio") or None
        cleaned_bio = clean_bio(raw_bio)

        bios_data.append({
            "username": username,
            "bio":      cleaned_bio,  # texto limpo; None se vazio após limpeza
        })

        status = f"{len(cleaned_bio.split())} tokens" if cleaned_bio else "vazia"
        print(f"bio coletada ({status})")

        # Checkpoint a cada 10 usuários
        if len(bios_data) % 10 == 0:
            pd.DataFrame(bios_data).to_csv(OUT_PATH, index=False)
            print(f"  [checkpoint] {OUT_PATH}")

except RateLimitException:
    print("\n[RATE LIMIT] Salvando progresso...")
finally:
    pd.DataFrame(bios_data).to_csv(OUT_PATH, index=False)

# =========================================================
# RESULTADO
# =========================================================

result_df  = pd.read_csv(OUT_PATH)
total      = len(result_df)
nao_vazias = result_df["bio"].notna().sum()
vazias     = total - nao_vazias

print(f"\nArquivo salvo       : {OUT_PATH}")
print(f"Total coletadas     : {total}")
print(f"Bios com conteúdo   : {nao_vazias}")
print(f"Bios vazias         : {vazias}")

if nao_vazias > 0:
    print(f"\nExemplos de bios limpas:")
    print(result_df[result_df["bio"].notna()][["username", "bio"]].head(5).to_string(index=False))