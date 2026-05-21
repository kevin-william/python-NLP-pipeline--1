# Fase 1: Pré-processamento e POS Tagging

Pipeline de processamento de linguagem natural para artigos da Wikipedia em português utilizando spaCy.

## Funcionalidades

- **Carregamento de Corpus**: Leitura e parsing de artigos da Wikipedia com metadados (título, URL)
- **Pré-processamento**: Tokenização, remoção de stopwords, lematização
- **POS Tagging**: Análise morfossintática completa com spaCy (`pt_core_news_lg`)
- **NER**: Reconhecimento de entidades nomeadas
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
│   ├── artigos_anotacao_lg.csv   # DataFrame com anotações
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

### CSV com Anotações
`output/artigos_anotacao_lg.csv` — DataFrame contendo:
- `artigo_id`: ID único do artigo
- `token_id`: Posição do token no artigo
- `token`: Token original
- `pos`: POS tag (NOUN, VERB, ADJ, etc.)
- `tag`: Tag detalhada
- `lemma`: Lema do token
- `dep_rel`: Relação de dependência
- `head_token`: Token cabeça na árvore de dependência
- `entity`: Entidade nomeada (se aplicável)
- `entity_label`: Tipo da entidade (PER, LOC, ORG, etc.)
- `title`: Título do artigo
- `url`: URL do artigo

### Visualizações
- `wordcloud.png` — Nuvem de palavras (sem stopwords)
- `pos_distribution.png` — Gráfico de barras com distribuição de POS tags
- `freq_comparison.png` — Comparativo de frequência antes/depois da remoção de stopwords

### Análise de Vocabulário
`output/vocabulario_analise.json` — Métricas:
- `vocab_raw_count`: Total de palavras únicas (com stopwords)
- `vocab_filtered_count`: Total de palavras únicas (sem stopwords)
- `vocab_reduction_percent`: Percentual de redução
- `top_20_raw`: 20 palavras mais frequentes (com stopwords)
- `top_20_filtered`: 20 palavras mais frequentes (sem stopwords)

## Configuração

As configurações estão em `src/fase1_config.py`:

```python
SPACY_MODEL = "pt_core_news_lg"    # Modelo spaCy
BATCH_SIZE = 10                     # Tamanho do batch para processamento
SEED = 42                           # Semente para reprodutibilidade
```

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

from corpus_loader import load_articles
from preprocessing import tokenize_article, get_stopwords
from pos_tagger import process_articles_batch

# Carregar artigos
articles = load_articles()

# Processar com spaCy (POS tagging completo)
df = process_articles_batch(articles)

# Analisar vocabulário
stopwords = get_stopwords()
print(f"Total de stopwords: {len(stopwords)}")

# Salvar resultado
df.to_csv("output/artigos_anotacao_lg.csv", index=False)
```

## Notas

- O modelo `pt_core_news_lg` é grande (~500MB). Na primeira execução, pode levar alguns minutos para下载.
- O processamento em lote (`BATCH_SIZE`) otimiza o uso de memória.
- A saída desta fase (`artigos_anotacao_lg.csv`) serve como entrada para a **Fase 2**.