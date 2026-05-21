# Plano - Fase 2: Processamento de Tokens e Busca por Similaridade

---

## 0. Organizacao do Projeto por Fases

### 0.1 Estrutura de Pastas

Cada fase (1-5) tera sua propria estrutura isolada de input, config e output:

```
python-NLP-pipeline-#1/
├── fase1/                    # Fase 1: Pre-processamento e POS Tagging
│   ├── input/                # dados de entrada
│   │   └── artigos_wikipedia.txt
│   ├── config/
│   │   └── config.py         # configuracoes especificas da fase
│   ├── output/               # saidas da fase
│   │   ├── artigos_anotacao_lg.csv
│   │   ├── nlp_pipeline.log
│   │   ├── wordcloud.png
│   │   └── vocabulario_analise.json
│   ├── src/                  # codigo da fase
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── corpus_loader.py
│   │   ├── preprocessing.py
│   │   ├── pos_tagger.py
│   │   ├── vocab_analysis.py
│   │   ├── wordcloud_gen.py
│   │   └── logger.py
│   └── tests/
│       └── ...
│
├── fase2/                    # Fase 2: Embeddings e Busca por Similaridade
│   ├── input/                # entrada: csv da fase1
│   │   └── artigos_anotacao_lg.csv
│   ├── config/
│   │   └── config.py         # metodos, parametros, etc
│   ├── output/               # modelos,visualizacoes
│   │   ├── embeddings/
│   │   ├── tsne_plot.png
│   │   └── fase2_pipeline.log
│   ├── src/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── embedding_pipeline.py
│   │   ├── embedding_config.py
│   │   ├── search_interface.py
│   │   ├── vectorizers/
│   │   │   ├── __init__.py
│   │   │   ├── bow_vectorizer.py
│   │   │   ├── tfidf_vectorizer.py
│   │   │   └── word2vec_vectorizer.py
│   │   ├── similarity/
│   │   │   ├── __init__.py
│   │   │   └── cosine_search.py
│   │   └── visualization/
│   │       ├── __init__.py
│   │       └── tsne_plot.py
│   └── tests/
│       └── ...
│
├── fase3/                    # Fase 3: (a definir)
├── fase4/                    # Fase 4: (a definir)
├── fase5/                    # Fase 5: (a definir)
│
├── shared/                   # Codigo compartilhado entre fases
│   ├── __init__.py
│   └── utils.py              # funcoes auxiliares
│
├── requirements.txt
└── README.md
```

### 0.2 Regra de Execucao

- Cada fase executa de forma independente
- Output da fase N serve como input para fase N+1
- Configuracao centralizada por fase
- Logs especifica por fase em `faseN/output/faseN.log`

### 0.3 Migracao da Fase 1

Mover estrutura atual para `fase1/`:
- `artigos_wikipedia.txt` → `fase1/input/`
- `src/` → `fase1/src/`
- `tests/` → `fase1/tests/`
- `output/` → `fase1/output/`
- `main.py` → `fase1/src/main.py`
- `config.py` → `fase1/config/config.py`

---

## Posicao no Pipeline

A Fase 2 sera integrada apos a Etapa 4 (Distribuicao de POS Tags) no arquivo `main.py`:

```python
# Step 4: POS distribution
logger.info("[ETAPA 4] Distribuicao de POS Tags")
plot_pos_distribution(df)

# === FASE 2: Token Embedding e Busca por Similaridade ===
logger.info("[FASE 2] Processamento de Tokens e Busca por Similaridade")
from src.embedding_pipeline import run_embedding_pipeline

run_embedding_pipeline(df)
```

---

## 1. Estrutura de Modulos

```
src/
  embedding_pipeline.py    # Orquestrador principal
  vectorizers/
    __init__.py
    bow_vectorizer.py      # CountVectorizer (Bag-of-Words)
    tfidf_vectorizer.py   # TfidfVectorizer
    word2vec_vectorizer.py # Gensim Word2Vec
  similarity/
    __init__.py
    cosine_search.py       # Busca por similaridade de cosseno
  visualization/
    __init__.py
    tsne_plot.py           # Visualizacao t-SNE
  search_interface.py      # Interface CMD de busca
```

---

## 2. Configuracao centralizada

Criar `src/embedding_config.py`:

```python
EMBEDDING_METHODS = ["bow", "tfidf", "word2vec"]  # ordem configuravel
SIMILARITY_METRIC = "cosine"
TOP_K_RESULTS = 10
WORD2VEC_PARAMS = {"vector_size": 100, "window": 5, "min_count": 2, "epochs": 10}
TSNE_PARAMS = {"n_components": 2, "perplexity": 30, "n_iter": 1000}
```

---

## 3. Modulos de Vectorizers (Desacoplados)

### 3.1 BOW Vectorizer

**Arquivo:** `src/vectorizers/bow_vectorizer.py`

```python
class BowVectorizer:
    def __init__(self, max_features=None, min_df=1):
        self.vectorizer = CountVectorizer(max_features=max_features, min_df=min_df)
        self.fitted = False

    def fit_transform(self, documents: List[str]):
        self.fitted = True
        return self.vectorizer.fit_transform(documents)

    def transform(self, documents: List[str]):
        return self.vectorizer.transform(documents)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()
```

### 3.2 TF-IDF Vectorizer

**Arquivo:** `src/vectorizers/tfidf_vectorizer.py`

```python
class TfidfVectorizerWrapper:
    def __init__(self, max_features=None, min_df=1, norm='l2'):
        self.vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df, norm=norm)
        self.fitted = False

    def fit_transform(self, documents: List[str]):
        self.fitted = True
        return self.vectorizer.fit_transform(documents)

    def transform(self, documents: List[str]):
        return self.vectorizer.transform(documents)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()
```

### 3.3 Word2Vec Vectorizer

**Arquivo:** `src/vectorizers/word2vec_vectorizer.py`

```python
class Word2VecWrapper:
    def __init__(self, vector_size=100, window=5, min_count=2, epochs=10):
        self.params = {"vector_size": vector_size, "window": window,
                       "min_count": min_count, "epochs": epochs}
        self.model = None

    def fit_transform(self, sentences: List[List[str]]):
        self.model = Word2Vec(sentences=sentences, **self.params)
        return self

    def get_sentence_vector(self, tokens: List[str]) -> np.ndarray:
        vectors = [self.model.wv[w] for w in tokens if w in self.model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.params["vector_size"])
```

---

## 4. Modulo de Busca por Similaridade

**Arquivo:** `src/similarity/cosine_search.py`

```python
class CosineSearchEngine:
    def __init__(self, vectorizer, documents: List[str]):
        self.vectorizer = vectorizer
        self.doc_vectors = None
        self.documents = documents

    def fit(self, documents: List[str]):
        self.doc_vectors = self.vectorizer.fit_transform(documents)

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [
            {"index": idx, "score": float(similarities[idx]), "document": self.documents[idx]}
            for idx in top_indices
        ]
```

---

## 5. Visualizacao t-SNE

**Arquivo:** `src/visualization/tsne_plot.py`

```python
class TSNEVisualizer:
    def __init__(self, perplexity=30, n_iter=1000, random_state=42):
        self.tsne = TSNE(n_components=2, perplexity=perplexity,
                         n_iter=n_iter, random_state=random_state)

    def fit_transform(self, embeddings: np.ndarray):
        return self.tsne.fit_transform(embeddings)

    def plot(self, embeddings: np.ndarray, labels: List[str], output_path: str):
        # Scatter plot com labels
        # Salvar em output_path
```

---

## 6. Pipeline Orquestrador

**Arquivo:** `src/embedding_pipeline.py`

```python
class EmbeddingPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.vectorizers = {}
        self.search_engines = {}

    def run(self, df: pd.DataFrame):
        logger.info("[FASE 2] Preparando documentos...")
        documents = self._prepare_documents(df)

        for method in self.config["EMBEDDING_METHODS"]:
            logger.info(f"[FASE 2] Treinando {method}...")
            vectorizer = self._get_vectorizer(method)
            engine = CosineSearchEngine(vectorizer, documents)
            engine.fit(documents)
            self.search_engines[method] = engine
            logger.info(f"[FASE 2] {method} treinado com sucesso")

        # t-SNE (opcional)
        if self.config.get("ENABLE_TSNE", False):
            self._run_tsne()

        return self.search_engines
```

---

## 7. Interface de Busca CMD

**Arquivo:** `src/search_interface.py`

```python
def start_search_interface(search_engines: dict):
    print("=" * 50)
    print("Sistema de Busca por Similaridade")
    print("Metodos disponiveis:", list(search_engines.keys()))
    print("=" * 50)

    while True:
        query = input("\nDigite sua consulta (ou 'sair'): ").strip()
        if query.lower() == "sair":
            break

        method = input("Metodo (bow/tfidf/word2vec): ").strip().lower()
        top_k = int(input("Quantos resultados (padrao 10): ") or "10")

        if method not in search_engines:
            print("Metodo invalido")
            continue

        results = search_engines[method].search(query, top_k)
        for r in results:
            print(f"[Score: {r['score']:.4f}] {r['document'][:100]}...")
```

---

## 8. Fluxo de integracao no main.py

```python
from src.embedding_pipeline import EmbeddingPipeline
from src.embedding_config import EMBEDDING_CONFIG
from src.search_interface import start_search_interface

# Apos Step 4
logger.info("[FASE 2] Processamento de Tokens e Busca por Similaridade")
pipeline = EmbeddingPipeline(EMBEDDING_CONFIG)
search_engines = pipeline.run(df)

# Iniciar interface de busca
start_search_interface(search_engines)
```

---

## 9. Logs

Adicionar logging detalhado em cada modulo:
- Logger em arquivo: `output/embedding_pipeline.log`
- Logger no terminal: nivel INFO
- Formato: `[TIMESTAMP] [LEVEL] [MODULO] message`

---

## 10. Testes

**Arquivos em `tests/`:**
- `test_bow_vectorizer.py`
- `test_tfidf_vectorizer.py`
- `test_word2vec_vectorizer.py`
- `test_cosine_search.py`
- `test_embedding_pipeline.py`
- `test_search_interface.py`

Cada teste deve verificar:
- Fit/Transform
- Resultado da busca
- Edge cases (query vazia, documentos vazios)

---

## 11. Atualizacao README.md

Adicionar secao:

```
## Fase 2: Token Embedding e Busca

### Metodos disponiveis
- Bag-of-Words (CountVectorizer)
- TF-IDF (TfidfVectorizer)
- Word2Vec (Gensim)

### Visualizacao
- t-SNE para embeddings

### Uso
```bash
python main.py
# Apos processamento, digite consultas no prompt
```
```

---

## 12. Dependencias a adicionar

Adicionar ao `requirements.txt` (se existir):
- scikit-learn
- gensim
- plotly (ou matplotlib para t-SNE)

---

## Resumo da ordem de implementacao

**Fase 0: Reorganizacao (PRIORIDADE)**

1. Criar pasta `fase1/` com estrutura: `input/`, `config/`, `output/`, `src/`, `tests/`
2. Mover arquivos existentes para `fase1/`
3. Criar `shared/` para codigo compartilhado
4. Atualizar imports e paths

**Fase 2: Implementacao**

5. Criar pasta `fase2/` com mesma estrutura
6. `fase2/config/config.py` - Configuracao (EMBEDDING_METHODS, etc)
7. `fase2/src/vectorizers/bow_vectorizer.py` - Bag-of-Words
8. `fase2/src/vectorizers/tfidf_vectorizer.py` - TF-IDF
9. `fase2/src/vectorizers/word2vec_vectorizer.py` - Word2Vec
10. `fase2/src/similarity/cosine_search.py` - Busca por similaridade
11. `fase2/src/visualization/tsne_plot.py` - Visualizacao t-SNE
12. `fase2/src/embedding_pipeline.py` - Orquestrador
13. `fase2/src/search_interface.py` - Interface CMD
14. Integrar no `fase2/src/main.py` (recebe CSV da fase1 como input)
15. Testes em `fase2/tests/`
16. Atualizar README.md

---

## Observacoes

- Fases 3, 4 e 5 seguirao o mesmo pattern de estrutura
- Cada fase tera seu proprio `main.py` executavel
- Logging centralizado por fase em `faseN/output/faseN.log`