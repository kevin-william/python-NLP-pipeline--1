# NLP Pipeline — Wikipedia Articles (Português)

Pipeline modular de Processamento de Linguagem Natural sobre artigos da Wikipedia em português, construído com spaCy, scikit-learn e Gensim. O projeto é organizado em **fases independentes e encadeadas**: o output de cada fase alimenta o input da próxima.

---

## Sumário

1. [Visão Geral](#visão-geral)
2. [Instalação](#instalação)
3. [Formato do Corpus](#formato-do-corpus)
4. [Fase 1 — Pré-processamento e POS Tagging](#fase-1--pré-processamento-e-pos-tagging)
5. [Fase 2 — Embeddings e Busca por Similaridade](#fase-2--embeddings-e-busca-por-similaridade)
6. [Testes](#testes)
7. [Estrutura do Projeto](#estrutura-do-projeto)

---

## Visão Geral

```
artigos_wikipedia.txt
        │
        ▼
  [ FASE 1 ] ── POS Tagging, NER, WordCloud, Análise de Vocabulário
        │
        │  artigos_anotacao_lg.csv
        ▼
  [ FASE 2 ] ── BOW / TF-IDF / Word2Vec + Busca por Similaridade (CLI)
        │
        ▼
  [ FASE N ] ── (futuro)
```

---

## Instalação

### Pré-requisitos

- Python 3.8+
- pip

### Instalando as dependências

Você pode instalar as dependências de cada fase individualmente ou tudo de uma vez:

```bash
# Dependências globais (inclui todas as fases)
pip install -r requirements.txt

# Apenas Fase 1
pip install -r fase1/requirements.txt

# Apenas Fase 2
pip install -r fase2/requirements.txt
```

### Baixar o modelo spaCy

A Fase 1 requer o modelo de português grande do spaCy:

```bash
python -m spacy download pt_core_news_lg
```

---

## Formato do Corpus

O arquivo de entrada da Fase 1 (`fase1/input/artigos_wikipedia.txt`) deve seguir o formato delimitado abaixo. Cada artigo é envolvido por marcadores de início e fim:

```
===== ARTICLE START =====
Title: Inteligência Artificial
URL: https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial
=========================
A inteligência artificial (IA) é a inteligência similar à humana...
===== ARTICLE END =====

===== ARTICLE START =====
Title: Processamento de Linguagem Natural
URL: https://pt.wikipedia.org/wiki/Processamento_de_linguagem_natural
=========================
O processamento de linguagem natural (PLN) é uma subárea...
===== ARTICLE END =====
```

> O parser extrai automaticamente `Title`, `URL` e o corpo do artigo.

---

## Fase 1 — Pré-processamento e POS Tagging

**Localização:** `fase1/`

### O que faz

A Fase 1 processa o corpus bruto e produz um DataFrame anotado com informações linguísticas para cada token:

| Etapa | Descrição |
|-------|-----------|
| **1. Carregamento** | Lê e parseia `artigos_wikipedia.txt`, extrai título, URL e conteúdo de cada artigo |
| **2. POS Tagging** | Aplica o modelo `pt_core_news_lg` via `nlp.pipe()` em lotes de 10 artigos |
| **3. Análise de Vocabulário** | Compara o vocabulário antes e depois da remoção de stopwords |
| **4. Distribuição POS** | Gera gráfico de barras com a frequência de cada classe gramatical |
| **5. Comparativo de Frequência** | Plota as top-N palavras com/sem stopwords |
| **6. WordCloud** | Gera nuvem de palavras a partir dos lemas filtrados |

### Como executar

```bash
cd fase1/src
python main.py
```

> Os caminhos são resolvidos relativamente ao diretório `fase1/`, então **não é necessário ajustar nenhum path** antes de rodar.

### Configuração

Edite `fase1/src/fase1_config.py` para ajustar o comportamento:

```python
SPACY_MODEL = "pt_core_news_lg"   # modelo spaCy a usar
BATCH_SIZE  = 10                   # artigos por lote (ajuste conforme RAM disponível)
SEED        = 42                   # reprodutibilidade
```

### Input

| Arquivo | Descrição |
|---------|-----------|
| `fase1/input/artigos_wikipedia.txt` | Corpus de artigos no formato descrito acima |

### Output

Todos os arquivos são salvos em `fase1/output/`:

| Arquivo | Descrição |
|---------|-----------|
| `artigos_anotacao_lg.csv` | DataFrame com uma linha por token, colunas: `artigo_id`, `token_id`, `token`, `pos`, `tag`, `lemma`, `dep_rel`, `head_token`, `entity`, `entity_label`, `title`, `url` |
| `wordcloud.png` | Nuvem de palavras gerada a partir dos lemas (sem stopwords) |
| `pos_distribution.png` | Gráfico de barras com a distribuição das POS tags no corpus |
| `freq_comparison.png` | Comparativo das palavras mais frequentes antes e depois da filtragem de stopwords |
| `vocabulario_analise.json` | Métricas de vocabulário: total de tokens, tokens únicos, redução percentual após filtragem |
| `nlp_pipeline.log` | Log completo de execução com timestamps |

### Colunas do CSV gerado

```
artigo_id   – índice do artigo (0-based)
token_id    – posição do token no documento
token       – texto original do token
pos         – classe gramatical universal (NOUN, VERB, ADJ, ...)
tag         – etiqueta morfológica detalhada
lemma       – forma lematizada do token
dep_rel     – relação de dependência sintática (nsubj, obj, ...)
head_token  – token cabeça na árvore de dependência
entity      – texto da entidade nomeada (vazio se não for entidade)
entity_label– tipo da entidade (PER, ORG, LOC, DATE, ...)
title       – título do artigo de origem
url         – URL do artigo de origem
```

### Stopwords customizadas

Para adicionar palavras à lista de stopwords do modelo, use a função `add_custom_stopwords` de `fase1/src/preprocessing.py`:

```python
from preprocessing import add_custom_stopwords
add_custom_stopwords(["exemplo", "palavra", "123"])
```

---

## Fase 2 — Embeddings e Busca por Similaridade

**Localização:** `fase2/`

### O que faz

A Fase 2 consome o CSV produzido pela Fase 1 e treina três representações vetoriais dos documentos, oferecendo em seguida uma interface de busca interativa via terminal:

| Método | Implementação | Descrição |
|--------|--------------|-----------|
| **BOW** | `CountVectorizer` (scikit-learn) | Vetores de contagem de ocorrências por termo |
| **TF-IDF** | `TfidfVectorizer` (scikit-learn) | Pesos TF-IDF normalizados (L2) |
| **Word2Vec** | `Word2Vec` (Gensim) | Média dos vetores de palavras do documento |

Após o treinamento, o sistema gera uma visualização **t-SNE** dos embeddings e inicia a interface de busca por **similaridade de cosseno**.

### Pré-requisito

A Fase 2 depende do output da Fase 1. Copie (ou mova) o CSV gerado:

```bash
# Opção A: copiar manualmente
copy fase1\output\artigos_anotacao_lg.csv fase2\input\

# Opção B: o caminho padrão já aponta para fase2/input/artigos_anotacao_lg.csv
# — basta garantir que o arquivo está lá antes de rodar a Fase 2
```

### Como executar

```bash
cd fase2/src
python main.py
```

O pipeline executa automaticamente:
1. Carregamento e agrupamento do CSV por documento
2. Treinamento dos vetorizadores configurados em `EMBEDDING_METHODS`
3. Geração do gráfico t-SNE (se `ENABLE_TSNE = True`)
4. Abertura da interface de busca interativa

### Configuração

Edite `fase2/src/fase2_config.py` para controlar todos os aspectos da fase:

```python
# Métodos a treinar — remova ou reordene conforme necessário
EMBEDDING_METHODS = ["bow", "tfidf", "word2vec"]

# Número de resultados retornados por busca
TOP_K_RESULTS = 10

# Parâmetros do Bag-of-Words
BOW_PARAMS = {
    "max_features": 5000,
    "min_df": 1,
}

# Parâmetros do TF-IDF
TFIDF_PARAMS = {
    "max_features": 5000,
    "min_df": 1,
    "norm": "l2",
}

# Parâmetros do Word2Vec (Gensim)
WORD2VEC_PARAMS = {
    "vector_size": 100,
    "window": 5,
    "min_count": 1,
    "epochs": 30,
    "seed": 42,
}

# Visualização t-SNE
ENABLE_TSNE = True
TSNE_PARAMS = {
    "n_components": 2,
    "perplexity": 5,
    "n_iter": 1000,
    "random_state": 42,
}
```

### Input

| Arquivo | Descrição |
|---------|-----------|
| `fase2/input/artigos_anotacao_lg.csv` | CSV com anotações token-a-token gerado pela Fase 1 |

### Output

Todos os arquivos são salvos em `fase2/output/`:

| Arquivo | Descrição |
|---------|-----------|
| `tsne_plot.png` | Visualização 2D dos embeddings Word2Vec via t-SNE, com rótulos de documento |
| `fase2_pipeline.log` | Log detalhado do treinamento e buscas realizadas |

### Interface de Busca (CLI)

Após o treinamento, o terminal exibe um prompt interativo:

```
============================================================
  SISTEMA DE BUSCA TEXTUAL POR SIMILARIDADE
============================================================
  Metodos disponiveis: bow, tfidf, word2vec
  Comandos:
    <consulta>           - busca usando o primeiro metodo (bow)
    <metodo> <consulta>  - busca com metodo especifico
    sair                 - encerra
============================================================

> inteligencia artificial
Buscando com [bow]: 'inteligencia artificial'
--------------------------------------------------
  #1 [Score: 0.8721] Doc #2
      inteligência artificial sistemas especialistas...

  #2 [Score: 0.7340] Doc #5
      aprendizado de máquina redes neurais...
--------------------------------------------------
  2 resultados encontrados.

> tfidf redes neurais profundas
Buscando com [tfidf]: 'redes neurais profundas'
--------------------------------------------------
  #1 [Score: 0.9102] Doc #5
      deep learning redes convolucionais...
--------------------------------------------------

> word2vec processamento linguagem
Buscando com [word2vec]: 'processamento linguagem'
--------------------------------------------------
  ...

> sair
Encerrando busca...
```

**Sintaxe dos comandos:**

| Comando | Comportamento |
|---------|--------------|
| `<consulta>` | Busca com o primeiro método da lista (`EMBEDDING_METHODS[0]`) |
| `<metodo> <consulta>` | Busca com o método especificado (`bow`, `tfidf` ou `word2vec`) |
| `sair` | Encerra a interface |

O score exibido é a **similaridade de cosseno** entre o vetor da consulta e o vetor de cada documento (0 a 1, quanto maior melhor).

---

## Testes

Cada fase possui uma suíte de testes independente com pytest:

```bash
# Apenas Fase 1
pytest fase1/tests/ -v

# Apenas Fase 2
pytest fase2/tests/ -v

# Todas as fases
pytest fase1/tests/ fase2/tests/ -v
```

**Cobertura de testes:**

| Fase | Módulos testados |
|------|-----------------|
| Fase 1 | `corpus_loader`, `preprocessing`, `pos_tagger`, pipeline completo |
| Fase 2 | `bow_vectorizer`, `tfidf_vectorizer`, `word2vec_vectorizer`, `cosine_search`, `search_interface`, pipeline completo |

---

## Estrutura do Projeto

```
python-NLP-pipeline-#1/
│
├── requirements.txt              # dependências globais
├── README.md
│
├── shared/
│   ├── __init__.py
│   └── utils.py                  # utilitários compartilhados entre fases
│
├── fase1/                        # Fase 1: Pré-processamento e POS Tagging
│   ├── requirements.txt
│   ├── input/
│   │   └── artigos_wikipedia.txt # corpus de entrada
│   ├── output/                   # gerado após execução
│   │   ├── artigos_anotacao_lg.csv
│   │   ├── wordcloud.png
│   │   ├── pos_distribution.png
│   │   ├── freq_comparison.png
│   │   ├── vocabulario_analise.json
│   │   └── nlp_pipeline.log
│   ├── src/
│   │   ├── main.py               # ponto de entrada
│   │   ├── fase1_config.py       # configurações e paths
│   │   ├── corpus_loader.py      # parsing do corpus
│   │   ├── preprocessing.py      # stopwords e filtragem
│   │   ├── pos_tagger.py         # POS tagging e NER com spaCy
│   │   ├── vocab_analysis.py     # análise e gráficos de vocabulário
│   │   ├── wordcloud_gen.py      # geração de nuvem de palavras
│   │   └── logger.py             # configuração de logging
│   └── tests/
│       ├── test_corpus_loader.py
│       ├── test_preprocessing.py
│       ├── test_pos_tagger.py
│       └── test_pipeline.py
│
├── fase2/                        # Fase 2: Embeddings e Busca por Similaridade
│   ├── requirements.txt
│   ├── input/
│   │   └── artigos_anotacao_lg.csv  # CSV da Fase 1
│   ├── output/                      # gerado após execução
│   │   ├── tsne_plot.png
│   │   └── fase2_pipeline.log
│   ├── src/
│   │   ├── main.py                  # ponto de entrada
│   │   ├── fase2_config.py          # configurações e parâmetros
│   │   ├── embedding_pipeline.py    # orquestra vetorizadores e busca
│   │   ├── search_interface.py      # interface CLI interativa
│   │   ├── logger.py
│   │   ├── vectorizers/
│   │   │   ├── bow_vectorizer.py    # Bag-of-Words
│   │   │   ├── tfidf_vectorizer.py  # TF-IDF
│   │   │   └── word2vec_vectorizer.py # Word2Vec (Gensim)
│   │   ├── similarity/
│   │   │   └── cosine_search.py     # busca por similaridade de cosseno
│   │   └── visualization/
│   │       └── tsne_plot.py         # projeção t-SNE
│   └── tests/
│       ├── test_bow_vectorizer.py
│       ├── test_tfidf_vectorizer.py
│       ├── test_word2vec_vectorizer.py
│       ├── test_cosine_search.py
│       ├── test_search_interface.py
│       └── test_embedding_pipeline.py
│
├── fase3/                        # (futuro)
├── fase4/                        # (futuro)
└── fase5/                        # (futuro)
├── shared/                   # código compartilhado
├── requirements.txt
└── README.md
```
