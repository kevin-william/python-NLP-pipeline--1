# Fase 1: Pré-processamento e POS Tagging

Pipeline de processamento de linguagem natural para artigos da Wikipedia em português utilizando spaCy.

## Funcionalidades

- **Carregamento de Corpus**: Leitura e parsing de artigos da Wikipedia com metadados (título, URL)
- **Filtro de Tamanho**: Remove artigos com menos de `MINIMO_PALAVRAS_ARTIGO` palavras antes do processamento; o log registra quantos foram removidos
- **Normalização Textual**: Etapa dedicada — lowercase, remoção de caracteres especiais e excesso de espaços (acentos preservados)
- **Tokenização Flexível**: Suporte a quatro tipos configuráveis — `palavra`, `bigrama`, `trigrama` e `sentenca`
- **Métodos de Processamento**: Lematização (spaCy), stemming (Snowball/Porter) ou sem normalização
- **Execução Combinada**: Pipeline rodada para cada combinação método × tipo de tokenização; artefatos separados por sufixo
- **POS Tagging**: Análise morfossintática completa com spaCy (`pt_core_news_lg`)
- **NER**: Reconhecimento de entidades nomeadas
- **Remoção de Stopwords**: Stopwords padrão do spaCy + customizadas em tempo de execução + extras via `STOPWORDS_EXTRAS`
- **WordCloud**: Geração de nuvens de palavras personalizáveis
- **Análise de Vocabulário**: Comparação antes/depois da remoção de stopwords, gráficos de frequência e distribuição de POS tags

## Estrutura de Arquivos

```
fase1/
├── input/
│   └── artigos_wikipedia.txt      # Dados de entrada
├── src/
│   ├── main.py                   # Script principal
│   ├── fase1_config.py           # Configurações
│   ├── logger.py                  # Sistema de logs
│   ├── corpus_loader.py          # Carregamento do corpus
│   ├── preprocessing.py          # Tokenização, stopwords, lemmatização
│   ├── pos_tagger.py             # POS tagging com spaCy
│   ├── vocab_analysis.py         # Análise de vocabulário
│   ├── wordcloud_gen.py          # Geração de wordcloud
│   └── vectorizers/              # (reservado para fase2)
├── output/                       # Resultados gerados
│   ├── 100-artigos_anotacao_lg.parquet   # DataFrame com anotações
│   ├── wordcloud.png            # Nuvem de palavras
│   ├── pos_distribution.png     # Distribuição de POS tags
│   ├── freq_comparison.png      # Comparativo de frequência
│   ├── vocabulario_analise.json # Métricas de vocabulário
│   └── nlp_pipeline.log         # Log completo
└── tests/                        # Testes unitários
    ├── test_corpus_loader.py
    ├── test_preprocessing.py
    ├── test_pos_tagger.py
    └── test_pipeline.py
```

## Requisitos

- Python 3.8+
- spaCy >= 3.7.0
- pandas >= 2.0.0
- wordcloud >= 1.9.0
- matplotlib >= 3.7.0
- pytest >= 7.0.0

## Instalação

```bash
pip install -r requirements.txt
python -m spacy download pt_core_news_lg
```

## Como Executar

```bash
cd fase1/src
python main.py
```

## Saída do Pipeline

Ao executar `main.py`, o pipeline produz:

### Parâmetros do Parquet gerado
`output/100-artigos_anotacao_lg_{metodo}_{tipo}.parquet` — DataFrame contendo:
- `id_artigo`: ID único do artigo
- `id_token`: Posição do token/n-grama/sentença no artigo
- `token`: Texto do token original (ou n-grama/sentença completa)
- `pos`: POS tag (NOUN, VERB, ADJ... | `SENT` para tipo sentença)
- `tag`: Tag detalhada
- `lema`: Lema do token
- `processado`: Token após método configurado (`none`, `lemmatizacao`, `stemming`)
- `relacao_dependencia`: Relação de dependência
- `token_cabeca`: Token cabeça na árvore de dependência
- `entidade`: Entidade nomeada (se aplicável)
- `rotulo_entidade`: Tipo da entidade (PER, LOC, ORG, etc.)
- `tipo_tokenizacao`: Tipo usado nesta execução (`palavra`, `bigrama`, `trigrama`, `sentenca`)
- `titulo`: Título do artigo
- `url`: URL do artigo

### Visualizações
- `wordcloud-100-artigos_{metodo}_{tipo}.png` — Nuvem de palavras
- `pos_distribution_{metodo}_{tipo}.png` — Gráfico de barras com distribuição de POS tags
- `freq_comparison_{metodo}_{tipo}.png` — Comparativo de frequência antes/depois da remoção de stopwords

### Análise de Vocabulário
`output/vocabulario_analise-100-artigos_{metodo}_{tipo}.json` — Métricas:
- `quantidade_vocabulario_bruto`: Total de palavras únicas (com stopwords)
- `quantidade_vocabulario_filtrado`: Total de palavras únicas (sem stopwords)
- `percentual_reducao_vocabulario`: Percentual de redução
- `top_20_bruto`: 20 palavras mais frequentes (com stopwords)
- `top_20_filtrado`: 20 palavras mais frequentes (sem stopwords)

## Configuração

As configurações estão em `src/fase1_config.py`:

```python
MODELO_SPACY = "pt_core_news_lg"              # Modelo spaCy
TAMANHO_LOTE = 5                               # Tamanho do lote de processamento
SEED_ALEATORIO = 42                            # Semente para reprodutibilidade

# Eixo 1: métodos de normalização do token
METODOS_PROCESSAMENTO_TOKENS = ['lemmatizacao', 'stemming', 'none']

# Eixo 2: tipos de tokenização
# Valores possíveis: 'palavra', 'bigrama', 'trigrama', 'sentenca'
# Exemplo: ['palavra', 'bigrama'] gera uma execução completa para cada tipo
TIPOS_TOKENIZACAO = ['palavra']

# Filtro: artigos com menos de N palavras são removidos antes do processamento
MINIMO_PALAVRAS_ARTIGO = 40

# Stopwords adicionais além das padrão do spaCy
STOPWORDS_EXTRAS = []
```

> A pipeline é executada para **cada combinação** `METODOS_PROCESSAMENTO_TOKENS` × `TIPOS_TOKENIZACAO`. Com `['lemmatizacao', 'stemming']` e `['palavra', 'bigrama']`, são geradas 4 execuções com artefatos separados.

## Testes

```bash
cd fase1
pytest tests/ -v
```

## Logs

Os logs são salvos em:
- Arquivo: `output/nlp_pipeline.log` (detalhado, DEBUG)
- Terminal: informações principais (INFO)

Formato: `[TIMESTAMP] [LEVEL] message`

## Exemplo de Uso Programático

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from corpus_loader import carregar_artigos, filtrar_artigos_por_tamanho
from preprocessing import tokenizar_por_tipo, obter_stopwords
from pos_tagger import processar_lote_artigos

# Carregar e filtrar artigos
artigos = carregar_artigos()
artigos, removidos = filtrar_artigos_por_tamanho(artigos)  # remove artigos muito curtos

# Tokenizar por tipo e processar com spaCy
dataframe = processar_lote_artigos(
    artigos,
    metodo_processamento="lemmatizacao",
    tipo_tokenizacao="bigrama",
)

# Inspecionar resultado
print(dataframe[["token", "lema", "processado", "tipo_tokenizacao"]].head())

# Salvar resultado
dataframe.to_parquet("output/resultado_lemmatizacao_bigrama.parquet", index=False)
```

## Notas

- O modelo `pt_core_news_lg` é grande (~500MB). Na primeira execução, pode levar alguns minutos para baixar.
- O processamento em lote (`TAMANHO_LOTE`) otimiza o uso de memória.
- A saída desta fase (`100-artigos_anotacao_lg.parquet`) serve como entrada para a **Fase 2**.