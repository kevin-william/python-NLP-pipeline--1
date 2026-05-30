# Fase 4 — Extração de Informações e Grafo de Conhecimento

## Objetivo

Implementar o Requisito 4 do projeto (NER, extração de informação e grafo de
conhecimento), atendendo aos critérios de rubrica que ficaram pendentes nas
Fases 1-3. Esta fase consome o parquet anotado gerado pela Fase 1 e o artefato
`.lpf2` da Fase 2, aplicando técnicas de extração de informação, normalização
de entidades, construção de grafo de conhecimento e visualização com displaCy.

---

## Critérios de Rubrica Atendidos

| Rubrica | Como é atendido |
|---|---|
| 4.1 — Extração de entidades/padrões com spaCy ou regex | `extracao_padroes.py` aplica regex para emails, URLs, datas, CPFs; `ner_analysis.py` usa spaCy para NER |
| 4.2 — Normalização/agrupamento de entidades (Levenshtein) | `fuzzy_matching.py` usa distância de Levenshtein para agrupar variações de entidades |
| 4.3 — Grafo de conhecimento com NetworkX (≥20 nós) | `grafo_conhecimento.py` constrói grafo bipartido entidade-documento ou entidade-relação com ≥20 nós |
| 4.4 — Cálculo de centralidade | `grafo_conhecimento.py` calcula betweenness, degree e eigenvector centrality |
| 4.5 — Visualização do grafo | `visualizacao_grafo.py` renderiza o grafo com matplotlib/PyVis |
| 4.6 — Pergunta analítica usando o grafo | `main.py` responde "Quais entidades são mais centrais no corpus e por quê?" |
| 4.7 — displaCy para NER | `ner_analysis.py` gera visualização HTML com `displacy.render()` |

---

## Estrutura de Arquivos

```
fase4/
├── input/                              # parquet da Fase 1 e .lpf2 da Fase 2
├── output/
│   ├── plots/                          # gráficos e visualizações
│   ├── displacy/                       # renderizações HTML do displaCy
│   ├── grafo_edges.csv                 # arestas do grafo exportadas
│   ├── entidades_extraidas.csv         # entidades e padrões extraídos
│   ├── fuzzy_matches.csv               # resultado do Levenshtein
│   └── fase4_pipeline.log
├── src/
│   ├── __init__.py
│   ├── fase4_config.py                 # constantes e caminhos
│   ├── logger.py                       # sistema de log
│   ├── main.py                         # executar_fase4_principal()
│   ├── extracao_padroes.py             # regex: emails, URLs, datas, CPFs
│   ├── fuzzy_matching.py               # Levenshtein para entidades
│   ├── ner_analysis.py                 # NER spaCy + displaCy
│   ├── relacoes.py                     # extração de relações SVO
│   ├── grafo_conhecimento.py           # NetworkX: construção, centralidade
│   └── visualizacao_grafo.py           # matplotlib/PyVis para grafo
├── tests/
│   ├── __init__.py
│   ├── test_extracao_padroes.py
│   ├── test_fuzzy_matching.py
│   ├── test_ner_analysis.py
│   ├── test_relacoes.py
│   └── test_grafo_conhecimento.py
├── requirements.txt
├── readme.md
└── plan.md
```

---

## Módulos — Detalhamento

### `fase4_config.py`

```python
DIRETORIO_BASE
CAMINHO_PARQUET_ENTRADA       # parquet da Fase 1 (com colunas entidade, rotulo_entidade)
CAMINHO_ARTEFATO_FASE2        # artefato .lpf2 da Fase 2
DIRETORIO_SAIDA
CAMINHO_LOG
DIRETORIO_PLOTS
DIRETORIO_DISPLACY

# Regex
REGEX_EMAILS
REGEX_URLS
REGEX_DATAS
REGEX_CPFs

# Fuzzy matching
MAX_LEVENSHTEIN_DISTANCE = 2

# Grafo
MINIMO_FREQUENCIA_ENTIDADE = 3
NUMERO_MINIMO_NOGRAFOS = 20
TOP_CENTRALIDADE = 10
```

---

### `extracao_padroes.py`

Extração de padrões textuais com expressões regulares, alinhado à Aula 07 do
professor (linhas 609-625 do `compilado_codigos.txt`).

```python
def extrair_emails(texto: str) -> List[str]
def extrair_urls(texto: str) -> List[str]
def extrair_datas(texto: str) -> List[str]
def extrair_cpfs(texto: str) -> List[str]
def extrair_todos_padroes(texto: str) -> Dict[str, List[str]]
def extrair_padroes_corpus(documentos: List[str]) -> pd.DataFrame
```

**Regex utilizadas** (mesmas do professor):
- Email: `r"[^\s]+@[a-zA-Z0-9\.]+\.[a-zA-Z]+"`
- URL: `r"https?://(?:www\.)?[a-zA-Z0-9-_.]+\.[a-zA-Z]{2,}(?:/[^\s]*)?"`
- Data: `r"\b\d{2}/\d{2}/\d{4}\b"`
- CPF: `r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"`

**Saída**: DataFrame com colunas `documento_id`, `tipo_padrao`, `valor`, `posicao_inicio`, `posicao_fim`.

---

### `fuzzy_matching.py`

Distância de Levenshtein para agrupar variações de entidades, alinhado à
Aula 07 do professor (linhas 627-646 do `compilado_codigos.txt`).

```python
def calcular_distancia_levenshtein(str1: str, str2: str) -> int
def agrupar_entidades_fuzzy(entidades: List[str], max_distancia: int = 2) -> Dict[str, List[str]]
def encontrar_mais_proximo(entrada: str, candidatos: List[str]) -> Tuple[str, int]
def normalizar_entidades(entidades: List[str]) -> pd.DataFrame
```

**Lógica**:
1. Extrai todas as entidades nomeadas do parquet (NER da Fase 1).
2. Calcula matriz de distância de Levenshtein entre pares.
3. Agrupa entidades com distância ≤ `MAX_LEVENSHTEIN_DISTANCE`.
4. Retorna DataFrame com `entidade_original`, `entidade_agrupada`, `distancia`.

**Dependência**: `python-Levenshtein` (ou `rapidfuzz` como alternativa leve).

---

### `ner_analysis.py`

Análise e visualização de entidades nomeadas com spaCy e displaCy,
alinhado à Aula 07/08 do professor (linhas 648-792 do `compilado_codigos.txt`).

```python
def extrair_entidades_documento(doc) -> List[Dict]
def extrair_entidades_corpus(documentos: List[str], nlp) -> pd.DataFrame
def contar_entidades_por_tipo(entidades_df: pd.DataFrame) -> pd.DataFrame
def gerar_displacy_html(doc, caminho_saida: str)
def gerar_displacy_corpus(documentos: List[str], nlp, diretorio_saida: str)
```

**Funcionalidades**:
1. Extrai entidades (PERSON, ORG, GPE, DATE, etc.) de cada documento.
2. Conta frequência por tipo de entidade.
3. Gera renderização HTML com `displacy.render(doc, style="ent")` para amostra.
4. Salva HTMLs em `output/displacy/`.

---

### `relacoes.py`

Extração de relações entre entidades (triplas sujeito-verbo-objeto),
alinhado à Aula 07 do professor (linhas 665-712 do `compilado_codigos.txt`).

```python
def extrair_triplas_svo(doc) -> List[Dict]
def extrair_relacoes_documento(doc) -> List[Dict]
def extrair_relacoes_corpus(documentos: List[str], nlp) -> pd.DataFrame
def construir_arestas_grafo(relacoes_df: pd.DataFrame) -> pd.DataFrame
```

**Lógica** (baseada no professor):
1. Para cada sentença, identifica entidades com label `ORG`, `PERSON`, `GPE`.
2. Identifica o verbo principal da sentença (`token.pos_ == 'VERB'`).
3. Monta tripla: `(entidade1, verbo, entidade2)`.
4. Converte triplas em arestas para o grafo: `source`, `target`, `relation`.

---

### `grafo_conhecimento.py`

Construção e análise do grafo de conhecimento com NetworkX,
alinhado à Aula 07 do professor (linhas 703-712 do `compilado_codigos.txt`).

```python
class GrafoConhecimento:
    def __init__(self)
    def construir_grafo(self, arestas: pd.DataFrame) -> nx.Graph
    def adicionar_nos_entidades(self, entidades_df: pd.DataFrame)
    def calcular_centralidade(self) -> Dict[str, pd.DataFrame]
    def obter_nos_mais_centrais(self, top_n: int = 10) -> pd.DataFrame
    def obter_arestas_mais_frequentes(self, top_n: int = 10) -> pd.DataFrame
    def responder_pergunta_analitica(self) -> str
    def exportar_grafo(self, caminho: str)
```

**Regras de negócio**:
- **Nós**: entidades nomeadas (PERSON, ORG, GPE) com frequência ≥ `MINIMO_FREQUENCIA_ENTIDADE`.
- **Arestas**: relações extraídas (ou co-ocorrência no mesmo documento).
- **Pesos**: frequência da relação ou co-ocorrência.
- **Mínimo**: ≥20 nós no grafo (requisito da rubrica).
- **Centralidade**: betweenness, degree, eigenvector.
- **Pergunta analítica**: "Quais entidades são mais centrais no corpus e por que?"
  Resposta baseada nos nós com maior betweenness centrality.

---

### `visualizacao_grafo.py`

Visualização do grafo de conhecimento com matplotlib e/ou PyVis.

```python
def plotar_grafo_matplotlib(grafo: nx.Graph, caminho_saida: str, figsize=(16, 12))
def plotar_grafo_pyvis(grafo: nx.Graph, caminho_saida: str)
def plotar_distribuicao_centralidade(centralidade: pd.DataFrame, caminho_saida: str)
def plotar_top_entidades(entidades_df: pd.DataFrame, caminho_saida: str)
```

**Visualizações**:
1. **Grafo com matplotlib**: nós coloridos por tipo, arestas com labels de relação, tamanho proporcional à centralidade.
2. **Grafo interativo com PyVis** (se disponível): HTML interativo.
3. **Distribuição de centralidade**: gráfico de barras com top-10 entidades por centralidade.
4. **Top entidades**: gráfico de barras com entidades mais frequentes por tipo.

---

### `main.py`

```python
def executar_fase4_principal()
```

**Fluxo**:

1. **Carregar dados**: parquet da Fase 1 (com entidades) + artefato .lpf2 da Fase 2.
2. **Extração de padrões**: aplicar regex em todos os documentos; exportar `entidades_extraidas.csv`.
3. **NER e displaCy**: extrair entidades com spaCy; gerar HTMLs de displaCy; exportar contagem.
4. **Fuzzy matching**: normalizar entidades com Levenshtein; exportar `fuzzy_matches.csv`.
5. **Extração de relações**: extrair triplas SVO; construir arestas.
6. **Grafo de conhecimento**: construir grafo com NetworkX; validar ≥20 nós.
7. **Análise de centralidade**: calcular betweenness, degree, eigenvector.
8. **Pergunta analítica**: responder "Quais entidades são mais centrais?"
9. **Visualizações**: grafo, distribuição de centralidade, top entidades.
10. **Exportar arestas**: `grafo_edges.csv`.

---

## Testes — Regras de Negócio

| Arquivo de teste | O que valida |
|---|---|
| `test_extracao_padroes.py` | regex extraem emails, URLs, datas e CPFs corretamente de textos de teste; retornam listas vazias quando não há padrões |
| `test_fuzzy_matching.py` | Levenshtein retorna distância correta; agrupamento funciona para entidades similares; `encontrar_mais_proximo` retorna a correspondência correta |
| `test_ner_analysis.py` | entidades são extraídas corretamente; contagem por tipo está correta; displaCy gera HTML válido |
| `test_relacoes.py` | triplas SVO são extraídas de sentenças simples; arestas são construídas corretamente |
| `test_grafo_conhecimento.py` | grafo construído tem ≥20 nós; centralidade é calculada; pergunta analítica retorna string não vazia |

---

## Dependências (`requirements.txt`)

```
networkx>=3.1
matplotlib>=3.7.0
pandas>=2.0.0
spacy>=3.7.0
python-Levenshtein>=0.21.0
pyvis>=0.3.2
```

---

## Como Executar

```bash
# A partir da raiz do projeto
python -m fase4.src.main

# Ou diretamente
cd fase4/src
python main.py

# Testes
python -m pytest fase4/tests -q
```

---

## Saídas Esperadas

| Arquivo | Descrição |
|---|---|
| `output/fase4_pipeline.log` | log detalhado da execução |
| `output/entidades_extraidas.csv` | entidades e padrões extraídos do corpus |
| `output/fuzzy_matches.csv` | resultado da normalização fuzzy das entidades |
| `output/grafo_edges.csv` | arestas do grafo de conhecimento |
| `output/plots/grafo_conhecimento.png` | visualização do grafo com matplotlib |
| `output/plots/centralidade_entidades.png` | distribuição de centralidade das entidades |
| `output/plots/top_entidades_por_tipo.png` | top entidades frequentes por tipo |
| `output/displacy/amostra_*.html` | renderizações HTML de NER com displaCy |

---

## Alinhamento com o Código do Professor

| Conceito (Aula 07/08) | Linhas do compilado | Implementação no projeto |
|---|---|---|
| Regex para emails, URLs, datas, CPFs | 609-625 | `extracao_padroes.py` |
| Distância de Levenshtein | 627-646 | `fuzzy_matching.py` |
| NER com spaCy | 648-663 | `ner_analysis.py` |
| Extração de relações (entidade-verbo-entidade) | 665-700 | `relacoes.py` |
| Grafo com arestas source-target | 703-706 | `grafo_conhecimento.py` |
| Centralidade/frequência de entidades | 708-712 | `grafo_conhecimento.py` |
| displaCy para visualização NER | 776-792 | `ner_analysis.py` |

---

## Justificativa Técnica

### Por que Levenshtein e não outro método?

O professor usa explicitamente a biblioteca `Levenshtein` nas aulas (Aula 07,
linha 627). Manter consistência com o conteúdo didático. Alternativas como
`fuzzywuzzy` ou `rapidfuzz` são equivalentes, mas `python-Levenshtein` é a
mais direta e rápida para este caso de uso.

### Por que NetworkX e não PyVis como padrão?

O professor usa `matplotlib` para visualizações estáticas. NetworkX + matplotlib
é a combinação mais simples e reproduzível. PyVis é uma extensão opcional para
visualização interativa (HTML), mas não deve ser a dependência principal.

### Por que extrair triplas SVO em vez de apenas co-ocorrência?

A rubrica pede "extração de relações ou triplas sujeito-verbo-objeto" (item 4.3).
Apenas co-ocorrência seria mais fraco. A abordagem do professor (linhas 684-700)
identifica entidades + verbo principal, o que atende melhor ao requisito.

### Validação do grafo (≥20 nós)

Se o corpus não gerar entidades suficientes para ≥20 nós, o pipeline deve:
1. Logar aviso com o número de nós encontrados.
2. Relaxar o filtro `MINIMO_FREQUENCIA_ENTIDADE` automaticamente.
3. Se ainda assim não atingir 20, logar erro e continuar (a análise fica parcial).
