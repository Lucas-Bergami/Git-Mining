# GitHub KDD — Descoberta de Conhecimento em Dados do GitHub

Trabalho prático de KDD (Knowledge Discovery in Databases) utilizando dados coletados da API pública do GitHub. O objetivo é percorrer as etapas da metodologia KDD — coleta, pré-processamento, análise exploratória, mineração e avaliação — aplicadas a uma rede social real.

---

## Estrutura dos Arquivos

```
├── user-data-mining.py       # Coleta de dados dos usuários via API do GitHub
├── repositoryMining.py       # Coleta de dados dos repositórios via API do GitHub
├── correlationTable.py       # Construção do dataset relacional usuário × repositório
├── repositoryKDD.py          # Análise exploratória dos repositórios (estatísticas, PCA, correlação)
├── user-csv-analyzer.py      # Análise exploratória dos usuários (estatísticas, histogramas, boxplots)
├── Modeltraining.py          # Treinamento e avaliação dos modelos de classificação
└── README.md
```

---

## Descrição dos Scripts

### `user-data-mining.py`
Coleta dados de usuários do GitHub a partir de um conjunto de usuários-semente (`torvalds`, `karpathy`, `tensorflow`, `pytorch`, `huggingface`), expandindo para seus seguidores. Para cada usuário, coleta atributos de perfil, atividade recente, linguagens utilizadas e engajamento. Salva checkpoints em CSV para retomada em caso de interrupção por rate limit.

**Saída:** `data/raw/users.csv`

---

### `repositoryMining.py`
Coleta dados dos repositórios pertencentes aos usuários coletados. Filtra repositórios por qualidade mínima (estrelas e tamanho) e coleta linguagens, tópicos, contribuidores e métricas temporais. Também salva checkpoints e trata o rate limit da API.

**Saída:** `data/raw/repos.csv`

---

### `correlationTable.py`
Constrói o dataset relacional cruzando usuários e repositórios. Gera pares positivos (usuário é dono do repositório) e pares negativos balanceados, priorizando negativos difíceis (usuário compartilha linguagens com o repositório mas não contribui). Calcula features relacionais como similaridade de Jaccard, diferenças de popularidade e compatibilidade tecnológica.

**Entrada:** `data/raw/users.csv`, `data/raw/repos.csv`  
**Saída:** `data/processed/user_repo_dataset.csv`

---

### `repositoryKDD.py`
Realiza a análise exploratória do dataset de repositórios. Gera estatísticas descritivas, histogramas, boxplots, matriz de correlação e projeção PCA dos dados.

**Entrada:** `data/raw/repos.csv`  
**Saída:** `data/analysis/` (estatísticas, gráficos, PCA)

---

### `user-csv-analyzer.py`
Realiza a análise exploratória do dataset de usuários. Gera estatísticas unificadas (numéricas e categóricas), histogramas, boxplots, gráficos de frequência e matriz de correlação.

**Entrada:** `data/raw/users.csv`  
**Saída:** `data/analysis/` (estatísticas, gráficos)

---

### `Modeltraining.py`
Treina e avalia dois modelos de classificação supervisionada — Random Forest e Regressão Logística — sobre o dataset relacional. Gera métricas de avaliação, matrizes de confusão, gráficos comparativos e importância de features.

**Entrada:** `data/processed/user_repo_dataset.csv`  
**Saída:** `data/models/plots/`, `data/models/reports/`

---

## Ordem de Execução

```bash
# 1. Coletar usuários
python user-data-mining.py

# 2. Coletar repositórios
python repositoryMining.py

# 3. Construir dataset relacional
python correlationTable.py

# 4. Análise exploratória dos usuários
python user-csv-analyzer.py data/raw/users.csv

# 5. Análise exploratória dos repositórios
python repositoryKDD.py

# 6. Treinar e avaliar modelos
python Modeltraining.py
```

---

## Requisitos

```bash
pip install requests pandas numpy matplotlib seaborn scikit-learn python-dotenv
```

---

## Configuração

Crie um arquivo `.env` na raiz do projeto com o seu token do GitHub:

```
GITHUB_TOKEN=seu_token_aqui
```

Para gerar um token: [github.com/settings/tokens](https://github.com/settings/tokens)  
O token precisa apenas de permissão de leitura pública (`public_repo`).

---

## Estrutura de Dados Gerada

```
data/
├── raw/
│   ├── users.csv                  # Dados dos usuários coletados
│   └── repos.csv                  # Dados dos repositórios coletados
├── checkpoints/
│   ├── all_users.csv              # Checkpoint da lista de usuários
│   └── users_data.csv             # Checkpoint dos dados de usuários
├── processed/
│   └── user_repo_dataset.csv      # Dataset relacional final
├── analysis/
│   ├── estatisticas.csv           # Estatísticas descritivas unificadas
│   ├── histogramas/               # Histogramas por atributo
│   ├── boxplots/                  # Boxplots por atributo
│   ├── categoricos/               # Gráficos de frequência categórica
│   └── correlacao.png             # Matriz de correlação
└── models/
    ├── plots/                     # Gráficos de avaliação dos modelos
    └── reports/                   # Relatórios e métricas dos modelos
```
