import re
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

import nltk

# ==========================================
# DOWNLOADS
# ==========================================

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# ==========================================
# CONFIG
# ==========================================

REPO_INPUT = "repos_textos.csv"
USER_INPUT = "users_texts.csv"

REPO_CLEAN = "repos_textos_limpos.csv"
USER_CLEAN = "users_textos_limpos.csv"

REPO_TFIDF = "repos_tfidf.csv"
USER_TFIDF = "user_tfidf.csv"

MAX_FEATURES = 5000


# ==========================================
# NLP
# ==========================================

stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()


# ==========================================
# LIMPEZA
# ==========================================


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"http\S+", " ", text)

    text = re.sub(r"\[.*?\]\(.*?\)", " ", text)

    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(r"\d+", " ", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    tokens = text.split()

    tokens = [
        lemmatizer.lemmatize(t) for t in tokens if (t not in stop_words and len(t) > 2)
    ]

    return " ".join(tokens)


# ==========================================
# CARREGAR DADOS
# ==========================================

repos = pd.read_csv(REPO_INPUT)

users = pd.read_csv(USER_INPUT)


# ==========================================
# LIMPAR
# ==========================================

print("Limpando repositórios...")

repos["clean_text"] = repos["full_text"].fillna("").apply(clean_text)

print("Limpando usuários...")

users["clean_text"] = users["text"].fillna("").apply(clean_text)


# ==========================================
# SALVAR LIMPOS
# ==========================================

repos.to_csv(REPO_CLEAN, index=False)

users.to_csv(USER_CLEAN, index=False)


# ==========================================
# TF-IDF COMPARTILHADO
# ==========================================

print("Treinando TF-IDF")

all_texts = pd.concat([repos["clean_text"], users["clean_text"]], ignore_index=True)

vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, min_df=2, max_df=0.90)

vectorizer.fit(all_texts)


# ==========================================
# TRANSFORMAR
# ==========================================

repo_matrix = vectorizer.transform(repos["clean_text"])

user_matrix = vectorizer.transform(users["clean_text"])


# ==========================================
# DATAFRAMES
# ==========================================

features = vectorizer.get_feature_names_out()


repo_df = pd.DataFrame(repo_matrix.toarray(), columns=features)

repo_df.insert(0, "repo_name", repos["repo_name"])


user_df = pd.DataFrame(user_matrix.toarray(), columns=features)

user_df.insert(0, "username", users["username"])


# ==========================================
# SALVAR
# ==========================================

repo_df.to_csv(REPO_TFIDF, index=False)

user_df.to_csv(USER_TFIDF, index=False)


# ==========================================
# RESUMO
# ==========================================

print("\nFinalizado\n")

print(f"Repos: {len(repos)}")

print(f"Users: {len(users)}")

print(f"Vocabulário: {len(features)}")

print(f"Repo matrix: {repo_matrix.shape}")

print(f"User matrix: {user_matrix.shape}")
