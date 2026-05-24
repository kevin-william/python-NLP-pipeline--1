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

### Parquet com Anotações
`output/100-artigos_anotacao_lg[_metodo].parquet` — DataFrame contendo:
- `id_artigo`: ID único do artigo
- `id_token`: Posição do token no artigo
- `token`: Token original
- `pos`: POS tag (NOUN, VERB, ADJ, etc.)
- `tag`: Tag detalhada
- `lema`: Lema do token
- `processado`: Token após método configurado (`none`, `lemmatizacao`, `stemming`)
- `relacao_dependencia`: Relação de dependência
- `token_cabeca`: Token cabeça na árvore de dependência
- `entidade`: Entidade nomeada (se aplicável)
- `rotulo_entidade`: Tipo da entidade (PER, LOC, ORG, etc.)
- `titulo`: Título do artigo
- `url`: URL do artigo

### Visualizações
- `wordcloud-100-artigos[_metodo].png` — Nuvem de palavras
- `pos_distribution[_metodo].png` — Gráfico de barras com distribuição de POS tags
- `freq_comparison[_metodo].png` — Comparativo de frequência antes/depois da remoção de stopwords

### Análise de Vocabulário
`output/vocabulario_analise-100-artigos[_metodo].json` — Métricas:
- `quantidade_vocabulario_bruto`: Total de palavras únicas (com stopwords)
- `quantidade_vocabulario_filtrado`: Total de palavras únicas (sem stopwords)
- `percentual_reducao_vocabulario`: Percentual de redução
- `top_20_bruto`: 20 palavras mais frequentes (com stopwords)
- `top_20_filtrado`: 20 palavras mais frequentes (sem stopwords)

## Configuração

As configurações estão em `src/fase1_config.py`:

```python
MODELO_SPACY = "pt_core_news_lg"              # Modelo spaCy
TAMANHO_LOTE = 30                              # Tamanho do lote de processamento
SEED_ALEATORIO = 42                            # Semente para reprodutibilidade
METODOS_PROCESSAMENTO_TOKENS = ['lemmatizacao']
STOPWORDS_EXTRAS = []
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

from corpus_loader import carregar_artigos
from preprocessing import tokenizar_artigo, obter_stopwords
from pos_tagger import processar_lote_artigos

# Carregar artigos
artigos = carregar_artigos()

# Processar com spaCy (POS tagging completo)
dataframe = processar_lote_artigos(artigos, metodo_processamento="lemmatizacao")

# Analisar vocabulário
stopwords_atuais = obter_stopwords()
print(f"Total de stopwords: {len(stopwords_atuais)}")

# Salvar resultado
dataframe.to_parquet("output/100-artigos_anotacao_lg.parquet", index=False)
```

## Notas

- O modelo `pt_core_news_lg` é grande (~500MB). Na primeira execução, pode levar alguns minutos para baixar.
- O processamento em lote (`TAMANHO_LOTE`) otimiza o uso de memória.
- A saída desta fase (`100-artigos_anotacao_lg.parquet`) serve como entrada para a **Fase 2**.