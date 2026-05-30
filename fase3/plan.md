# Fase 3 — Modelagem de Tópicos (LSA, LDA, NMF)

## Objetivo

Consumir o artefato `.lpf2` exportado pela Fase 2 e executar três técnicas de
modelagem de tópicos sobre o corpus de artigos da Wikipedia, atendendo aos
critérios:

1. **EDA textual** — caracterizar distribuição de comprimento dos documentos,
   tamanho do vocabulário, frequência de termos e desequilíbrios do corpus.
2. **Modelagem de tópicos** — LSA, LDA e NMF, cada um coerente com as
   representações matriciais disponíveis no artefato.
3. **Comparação explícita** — métricas de coerência, perplexidade (LDA) e
   visualizações comparativas entre os três algoritmos.

---

## Critérios de Avaliação Atendidos

| Critério | Como é atendido |
|---|---|
| EDA textual com caracterização do corpus | `eda.py` gera estatísticas de comprimento, vocabulário, top-frequências e gráficos salvos em `output/plots/` |
| Técnicas coerentes com o corpus | TF-IDF matrix (do artefato) alimenta LSA e NMF; matriz count alimenta LDA |
| Comparação de algoritmos com métricas e visualizações | `avaliacao.py` tabela perplexidade (LDA), coerência top-N (todos), heatmaps de tópico×palavra |

---

## Estrutura de Arquivos

```
fase3/
├── input/                          # artefato .lpf2 copiado da fase2
├── output/
│   ├── plots/                      # gráficos EDA e visualizações de tópicos
│   └── fase3_pipeline.log
├── src/
│   ├── __init__.py
│   ├── fase3_config.py             # constantes e caminhos
│   ├── logger.py                   # mesmo padrão das fases anteriores
│   ├── main.py                     # executar_fase3_principal()
│   ├── carregador_artefato.py      # carrega ArtifactFase2 do .lpf2
│   ├── eda.py                      # análise exploratória textual
│   ├── avaliacao.py                # métricas e comparação entre modelos
│   ├── visualizacao.py             # heatmaps, distribuições tópico-documento
│   └── modelos_topicos/
│       ├── __init__.py
│       ├── lsa_modelo.py           # LSA via TruncatedSVD + TF-IDF
│       ├── lda_modelo.py           # LDA via sklearn
│       └── nmf_modelo.py           # NMF via sklearn
├── tests/
│   ├── __init__.py
│   ├── test_carregador_artefato.py
│   ├── test_eda.py
│   ├── test_lsa.py
│   ├── test_lda.py
│   ├── test_nmf.py
│   ├── test_avaliacao.py
│   └── test_pipeline.py
├── requirements.txt
├── README.md
└── plan.md
```

---

## Módulos — Detalhamento

### `fase3_config.py`

Constantes centralizadas seguindo o padrão das fases anteriores (nomes em
português, sem magic strings no código):

```python
CAMINHO_ARTEFATO_FASE2     # caminho para o .lpf2 em output/artifacts/
DIRETORIO_SAIDA
CAMINHO_LOG
DIRETORIO_PLOTS
NUM_TOPICOS                # padrão: 10
TOP_N_PALAVRAS             # top palavras exibidas por tópico: 10
PARAMS_LDA                 # max_iter, random_state, etc.
PARAMS_NMF                 # init, random_state, etc.
PARAMS_LSA                 # n_components, random_state, etc.
```

---

### `carregador_artefato.py`

Responsabilidade única: desserializar o `ArtifactFase2` do arquivo `.lpf2`.

```python
def carregar_artefato_fase2(caminho: str) -> ArtifactFase2
```

- Valida existência do arquivo.
- Valida tipo do objeto desserializado.
- Valida que `documentos`, `tfidf_matrix` e `tfidf_vectorizer` não são `None`.
- Loga estatísticas básicas (nº de documentos, shape da matriz).

---

### `eda.py`

EDA textual completa sobre os documentos do artefato:

```python
def executar_eda(documentos: List[str], titulos: List[str], diretorio_saida: str) -> dict
```

Análises realizadas:
- **Comprimento dos documentos**: histograma de número de tokens por documento;
  estatísticas (média, mediana, desvio padrão, min, max).
- **Tamanho do vocabulário**: total de termos únicos no corpus.
- **Distribuição de frequência de termos**: curva de Zipf, top-20 termos mais
  frequentes (gráfico de barras).
- **Desequilíbrio**: comprimento relativo entre documentos (box plot).
- Retorna `dict` com todas as métricas numéricas para uso nos testes.

Gráficos salvos em `output/plots/eda_*.png`.

---

### `modelos_topicos/lsa_modelo.py`

```python
class ModeloLSA:
    def __init__(self, num_topicos: int, params: dict)
    def treinar(self, matriz_tfidf) -> ModeloLSA
    def obter_topicos(self, vocabulario: List[str], top_n: int) -> List[List[str]]
    def obter_distribuicao_documentos(self) -> np.ndarray
```

Implementação:
- Usa `sklearn.decomposition.TruncatedSVD` sobre a matriz TF-IDF do artefato.
- `treinar()` ajusta o modelo e armazena `self.matriz_documento_topico`.
- `obter_topicos()` retorna as `top_n` palavras por componente.

---

### `modelos_topicos/lda_modelo.py`

```python
class ModeloLDA:
    def __init__(self, num_topicos: int, params: dict)
    def treinar(self, matriz_bow) -> ModeloLDA
    def obter_topicos(self, vocabulario: List[str], top_n: int) -> List[List[str]]
    def obter_distribuicao_documentos(self) -> np.ndarray
    def obter_perplexidade(self, matriz_bow) -> float
```

Implementação:
- Usa `sklearn.decomposition.LatentDirichletAllocation`.
- Requer a matriz BoW (count) do artefato (ou reconstrói via `CountVectorizer`
  se `bow_matrix` for `None`).
- Expõe `perplexidade` como métrica de avaliação.

---

### `modelos_topicos/nmf_modelo.py`

```python
class ModeloNMF:
    def __init__(self, num_topicos: int, params: dict)
    def treinar(self, matriz_tfidf) -> ModeloNMF
    def obter_topicos(self, vocabulario: List[str], top_n: int) -> List[List[str]]
    def obter_distribuicao_documentos(self) -> np.ndarray
```

Implementação:
- Usa `sklearn.decomposition.NMF` sobre a matriz TF-IDF.
- `treinar()` ajusta H (tópico×palavra) e W (documento×tópico).

---

### `avaliacao.py`

```python
def calcular_coerencia_top_n(topicos: List[List[str]], documentos: List[str]) -> float
def comparar_modelos(resultados: dict) -> pd.DataFrame
```

- `calcular_coerencia_top_n`: métrica de coerência simples via co-ocorrência
  das top-N palavras de cada tópico nos documentos (sem depender do Gensim).
- `comparar_modelos`: recebe dict `{"LSA": {coerencia, topicos}, "LDA": {...},
  "NMF": {...}}` e retorna DataFrame com colunas
  `modelo | num_topicos | coerencia_media | perplexidade`.
- O salvamento do CSV (`output/comparacao_modelos.csv`) é responsabilidade de `main.py`.

---

### `visualizacao.py`

```python
def plotar_top_palavras_por_topico(topicos, nome_modelo, diretorio_saida)
def plotar_distribuicao_topicos_documentos(distribuicao, titulos, nome_modelo, diretorio_saida)
def plotar_comparacao_metricas(dataframe_comparacao, diretorio_saida)
```

- `plotar_top_palavras_por_topico`: grade de subplots com barras horizontais,
  uma por tópico — salvo em `output/plots/{modelo}_topicos.png`.
- `plotar_distribuicao_topicos_documentos`: heatmap documento×tópico.
- `plotar_comparacao_metricas`: gráfico de barras comparando coerência média
  entre LSA, LDA e NMF.

---

### `main.py`

```python
def executar_fase3_principal()
```

Fluxo:
1. Carrega artefato via `carregador_artefato.carregar_artefato_fase2()`.
2. Executa EDA e salva relatório.
3. Treina LSA, LDA e NMF.
4. Extrai tópicos de cada modelo.
5. Calcula métricas de comparação.
6. Gera visualizações.
7. Loga sumário final com tabela comparativa.

---

## Testes — Regras de Negócio

| Arquivo de teste | O que valida |
|---|---|
| `test_carregador_artefato.py` | arquivo válido carrega; arquivo ausente lança `FileNotFoundError`; tipo errado lança `TypeError`; artefato com `documentos` vazio lança `ValueError` |
| `test_eda.py` | retorno contém todas as chaves esperadas; comprimento mínimo > 0; vocabulário > 0; gráficos criados no diretório |
| `test_lsa.py` | shape de `obter_distribuicao_documentos()` é `(n_docs, n_topicos)`; `obter_topicos()` retorna `num_topicos` listas com `top_n` strings |
| `test_lda.py` | mesmo que LSA + perplexidade é float positivo (>= 0) |
| `test_nmf.py` | mesmo que LSA; valores da distribuição são não-negativos |
| `test_avaliacao.py` | DataFrame de comparação tem 3 linhas (LSA, LDA, NMF) e colunas esperadas; coerência entre 0 e 1 |
| `test_pipeline.py` | integração: pipeline completo roda com artefato sintético sem erros |

---

## Dependências (`requirements.txt`)

```
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scipy>=1.11.0
joblib>=1.3.0
```

> Gensim **não** é necessário — coerência calculada localmente para manter
> dependências mínimas.

---

## Como Executar

```bash
# A partir da raiz do projeto
python -m fase3.src.main

# Ou diretamente
cd fase3/src
python main.py

# Testes
python -m pytest fase3/tests -q
```

---

## Saídas Esperadas

| Arquivo | Descrição |
|---|---|
| `output/fase3_pipeline.log` | log detalhado da execução |
| `output/comparacao_modelos.csv` | tabela comparativa LSA × LDA × NMF |
| `output/plots/eda_comprimento.png` | histograma de comprimento de documentos |
| `output/plots/eda_top_termos.png` | top-20 termos mais frequentes |
| `output/plots/eda_zipf.png` | curva de Zipf do corpus |
| `output/plots/eda_boxplot.png` | box plot de comprimento por documento |
| `output/plots/lsa_topicos.png` | top palavras por tópico — LSA |
| `output/plots/lda_topicos.png` | top palavras por tópico — LDA |
| `output/plots/nmf_topicos.png` | top palavras por tópico — NMF |
| `output/plots/lsa_heatmap.png` | heatmap documento×tópico — LSA |
| `output/plots/lda_heatmap.png` | heatmap documento×tópico — LDA |
| `output/plots/nmf_heatmap.png` | heatmap documento×tópico — NMF |
| `output/plots/comparacao_coerencia.png` | comparação de coerência entre modelos |
