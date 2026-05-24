# Fase 2: Token Embedding e Busca por Similaridade

Pipeline para geração de embeddings de texto e busca textual por similaridade.

## Funcionalidades

- **Bag-of-Words (BoW)**: Representação vetorial usando `CountVectorizer`
- **TF-IDF**: Representação vetorial com ponderação por frequência
- **Word2Vec**: Embeddings densos usando Gensim
- **Busca por Similaridade**: Interface interativa CMD para consultas textuais
- **t-SNE**: Visualização 2D dos embeddings
- **Fila de Métodos Configurável**: Escolha quais métodos usar e em qual ordem

## Estrutura de Arquivos

```
fase2/
├── input/
│   └── artigos_anotacao_lg.parquet   # Saída da Fase 1 (obrigatório)
├── src/
│   ├── main.py                   # Script principal
│   ├── fase2_config.py           # Configurações
│   ├── logger.py                 # Sistema de logs
│   ├── embedding_pipeline.py     # Orquestrador do pipeline
│   ├── search_interface.py      # Interface CMD de busca
│   ├── vectorizers/              # Implementações de vetorização
│   │   ├── bow_vectorizer.py
│   │   ├── tfidf_vectorizer.py
│   │   └── word2vec_vectorizer.py
│   ├── similarity/               # Busca por similaridade
│   │   └── cosine_search.py
│   └── visualization/             # Visualizações
│       └── tsne_plot.py
├── output/                       # Resultados gerados
│   ├── fase2_pipeline.log        # Log detalhado
│   └── tsne_plot.png            # Visualização t-SNE
└── tests/                        # Testes unitários
    ├── test_bow_vectorizer.py
    ├── test_tfidf_vectorizer.py
    ├── test_word2vec_vectorizer.py
    ├── test_cosine_search.py
    ├── test_embedding_pipeline.py
    └── test_search_interface.py
```

## Pré-requisitos

### 1. Executar a Fase 1 Primeiro

A Fase 2 depende do arquivo Parquet gerado pela Fase 1:

```bash
# 1. Executar Fase 1
cd fase1/src
python main.py

# 2. Copiar o Parquet para a Fase 2 (se não foi feito automaticamente)
copy fase1\output\100-artigos_anotacao_lg.parquet fase2\input\artigos_anotacao_lg.parquet
```

### Dependências

- Python 3.8+
- scikit-learn >= 1.3.0
- gensim >= 4.3.0
- numpy >= 1.24.0

```bash
pip install scikit-learn gensim
```

## Configuração

As configurações estão em `src/fase2_config.py`:

```python
# Métodos a serem treinados (ordem importa!)
METODOS_EMBEDDING = ["bow", "tfidf", "word2vec"]

# Parâmetros de cada método
PARAMS_BOW = {"max_features": 5000, "min_df": 1}
PARAMS_TFIDF = {"max_features": 5000, "min_df": 1, "norm": "l2"}
PARAMS_WORD2VEC = {
    "vector_size": 100,
    "window": 5,
    "min_count": 1,
    "epochs": 30,
    "seed": 42
}

# Configurações de busca
TOP_K_RESULTADOS = 10

# Visualização t-SNE
HABILITAR_TSNE = True
PARAMS_TSNE = {"n_components": 2, "perplexity": 30, "n_iter": 2000, "random_state": 42}
PARAMS_PLOT_TSNE = {
    "figsize": (16, 12),
    "dpi": 150,
    "marker_size": 50,
    "annotate_fontsize": 7,
}
```

### Exemplos de Configuração

**Usar apenas TF-IDF:**
```python
METODOS_EMBEDDING = ["tfidf"]
```

**Usar Word2Vec + TF-IDF:**
```python
METODOS_EMBEDDING = ["word2vec", "tfidf"]
```

**Aumentar dimensões do Word2Vec:**
```python
PARAMS_WORD2VEC = {"vector_size": 200, "window": 5, "min_count": 2, "epochs": 50}
```

### Parâmetros do t-SNE

| Parâmetro | Valor Padrão | Descrição |
|-----------|-------------|-----------|
| `perplexity` | 30 | Vizinhos considerados. Baixo (5-10) = clusters compactos. Alto (30-50) = distribuição uniforme |
| `n_iter` | 2000 | Iterações de otimização. Valores maiores melhoram o resultado |
| `n_components` | 2 | Dimensões do resultado (2 = 2D, 3 = 3D) |
| `random_state` | 42 | Semente para reprodutibilidade |

### Parâmetros do Plot (visualização)

| Parâmetro | Valor Padrão | Descrição |
|-----------|-------------|-----------|
| `figsize` | (16, 12) | Tamanho da figura em polegadas (largura, altura) |
| `dpi` | 150 | Resolução da imagem salva |
| `marker_size` | 50 | Tamanho dos pontos no scatter plot |
| `annotate_fontsize` | 7 | Tamanho da fonte das anotações |

### Valores Recomendados por Cenário

| Cenário | perplexity | n_iter | figsize |
|---------|------------|--------|---------|
| Poucos dados (<100 docs) | 5-10 | 1000 | (10, 8) |
| Dados médios (100-1000 docs) | 30-50 | 2000 | (14, 10) |
| Muitos dados (>1000 docs) | 50-100 | 3000 | (16, 12) |

### Exemplo: Visualização condensada

Se a visualização estiver com pontos muito aglomerados, aumente `perplexity` e `figsize`:
```python
PARAMS_TSNE = {"perplexity": 50, "n_iter": 3000, "random_state": 42}
PARAMS_PLOT_TSNE = {"figsize": (20, 14), "dpi": 200, "marker_size": 40}
```

## Como Executar

```bash
cd fase2/src
python main.py
```

### O que acontece:

1. **Carregamento**: O pipeline lê o Parquet da Fase 1 (`input/artigos_anotacao_lg.parquet`)
2. **Pré-processamento**: Constrói documentos a partir dos lemas (agrupados por id_artigo), remove pontuação
3. **Treinamento**: Treina cada método de embedding configurado
4. **t-SNE**: Gera visualização 2D dos embeddings (se habilitado)
5. **Interface de Busca**: Abre prompt interativo para consultas

## Interface de Busca CMD

Ao final do treinamento, o sistema entra em modo interativo:

```
============================================================
  SISTEMA DE BUSCA TEXTUAL POR SIMILARIDADE
============================================================
  Metodos disponiveis: bow, tfidf, word2vec
  Comandos:
    <consulta>           - busca usando o primeiro metodo
    <metodo> <consulta>  - busca com metodo especifico
    sair                 - encerra
============================================================
```

### Exemplos de Consulta

**Busca simples (usa primeiro método):**
```
> python processamento natural
```

**Busca com método específico:**
```
> bow aprendizado de maquina
> tfidf linguagem natural
> word2vec redes neurais
```

**Buscar com mais resultados:**
```
> 20 processamentos de linguagem natural
```

### Saída dos Resultados

```
Buscando com [tfidf]: 'processamento natural'
--------------------------------------------------
  #1 [Score: 0.8723] Doc #1
      processamento linguagem natural dados...

  #2 [Score: 0.6541] Doc #3
      linguagem natural processamento texto...

--------------------------------------------------
  2 resultados encontrados.
```

Cada resultado contém:
- `Score`: Similaridade de cosseno (0 a 1)
- `Doc #N`: ID do documento no corpus
- Preview: Primeiros 200 caracteres do documento

Para sair, digite `sair`.

## Uso Programático

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from embedding_pipeline import PipelineEmbeddings
from fase2_config import (
    CAMINHO_PARQUET_ENTRADA,
    METODOS_EMBEDDING,
    PARAMS_BOW,
    PARAMS_TFIDF,
    PARAMS_WORD2VEC,
    TOP_K_RESULTADOS,
    HABILITAR_TSNE,
    PARAMS_TSNE,
    PARAMS_PLOT_TSNE,
)

# Configuração
config = {
    "METODOS_EMBEDDING": METODOS_EMBEDDING,
    "PARAMS_BOW": PARAMS_BOW,
    "PARAMS_TFIDF": PARAMS_TFIDF,
    "PARAMS_WORD2VEC": PARAMS_WORD2VEC,
    "TOP_K_RESULTADOS": TOP_K_RESULTADOS,
    "HABILITAR_TSNE": HABILITAR_TSNE,
    "PARAMS_TSNE": PARAMS_TSNE,
    "PARAMS_PLOT_TSNE": PARAMS_PLOT_TSNE,
}

# Criar e executar pipeline
pipeline = PipelineEmbeddings(config, CAMINHO_PARQUET_ENTRADA)
search_engines = pipeline.executar()

# Consulta via código
results = pipeline.buscar_texto("bow", "inteligencia artificial", top_k=5)

for r in results:
    print(f"Score: {r['score']:.4f} | Doc #{r['index']}")
```

## Testes

```bash
cd fase2
pytest tests/ -v
```

### Cobertura

- `test_bow_vectorizer.py`: Fit, transform, edge cases
- `test_tfidf_vectorizer.py`: Fit, transform, valores positivos
- `test_word2vec_vectorizer.py`: Fit, sentence embeddings, palavras desconhecidas
- `test_cosine_search.py`: Busca, ordenação por score
- `test_embedding_pipeline.py`: Pipeline completo, múltiplos métodos
- `test_search_interface.py`: Imports e estrutura

## Diferenças entre Métodos

| Método | Tipo | Velocidade | Uso de Memória | Melhor para |
|--------|------|------------|----------------|-------------|
| **BoW** | Esparso | Rápido | Médio | Baseline, debug |
| **TF-IDF** | Esparso | Rápido | Médio | Busca textual, relevância |
| **Word2Vec** | Denso | Moderado | Baixo | Semântica, similaridade |

### Observações

- **BoW** e **TF-IDF** são métodos esparsos (matrizes com muitos zeros)
- **Word2Vec** gera vetores densos (representação contínua)
- A similaridade de cosseno funciona bem para todos
- t-SNE usa o primeiro método disponível para visualização

## Logs

Os logs são salvos em:
- Arquivo: `output/fase2_pipeline.log` (detalhado, DEBUG)
- Terminal: informações principais (INFO)

Formato: `[TIMESTAMP] [LEVEL] [MODULE] message`

## Fluxo de Dados

```
Fase 1 Output                    Fase 2 Input                Fase 2 Output
┌─────────────────┐             ┌─────────────────┐          ┌─────────────────┐
│ artigos_anotacao│ ─────────> │ artigos_anotacao│          │ tsne_plot.png   │
│ _lg.parquet     │            │ _lg.parquet     │          │ fase2.log       │
│ (tokens+POS)    │            │ (tokens+POS)    │          │                 │
└─────────────────┘            └─────────────────┘          └─────────────────┘
                                       │
                                       v
                              ┌─────────────────┐
                              │ EmbeddingPipeline
                              │ - tokenização   │
                              │ - vetorização   │
                              │ - busca         │
                              └─────────────────┘
```

## Notas

- O CSV de entrada deve conter as colunas: `artigo_id`, `lemma`, `pos`, `title`
- A busca por similaridade usa similaridade de cosseno entre vetores
- O Word2Vec precisa de ao menos uma ocorrência de cada palavra (`min_count=1` por padrão)
- Para textos pequenos, o TF-IDF geralmente oferece melhores resultados
- Para capturar relações semânticas, o Word2Vec é mais adequado