import os
import random
import numpy as np

SEED_ALEATORIO = 42
random.seed(SEED_ALEATORIO)
np.random.seed(SEED_ALEATORIO)

DIRETORIO_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAMINHO_ENTRADA = os.path.join(DIRETORIO_BASE, "input", "artigos_wikipedia_100_formatado.txt")
DIRETORIO_SAIDA = os.path.join(DIRETORIO_BASE, "output")
CAMINHO_LOG = os.path.join(DIRETORIO_SAIDA, "nlp_pipeline-100-artigos.log")
CAMINHO_PARQUET_SAIDA = os.path.join(DIRETORIO_SAIDA, "100-artigos_anotacao_lg.parquet")
CAMINHO_NUVEM_PALAVRAS = os.path.join(DIRETORIO_SAIDA, "wordcloud-100-artigos.png")
CAMINHO_ANALISE_VOCABULARIO = os.path.join(DIRETORIO_SAIDA, "vocabulario_analise-100-artigos.json")

# CAMINHO_ENTRADA = os.path.join(DIRETORIO_BASE, "input", "artigos_wikipedia.txt")
# DIRETORIO_SAIDA = os.path.join(DIRETORIO_BASE, "output")
# CAMINHO_LOG = os.path.join(DIRETORIO_SAIDA, "nlp_pipeline.log")
# CAMINHO_PARQUET_SAIDA = os.path.join(DIRETORIO_SAIDA, "anotacao_lg.parquet")
# CAMINHO_NUVEM_PALAVRAS = os.path.join(DIRETORIO_SAIDA, "wordcloud.png")
# CAMINHO_ANALISE_VOCABULARIO = os.path.join(DIRETORIO_SAIDA, "vocabulario_analise.json")

MODELO_SPACY = "pt_core_news_lg"
TAMANHO_LOTE = 5

MARCADOR_INICIO_ARTIGO = "===== ARTICLE START ====="
MARCADOR_FIM_ARTIGO = "===== ARTICLE END ====="

# Métodos de processamento de tokens: 'none', 'lemmatizacao', 'stemming'
# Pode ser uma lista com um ou múltiplos valores para execução sequencial
METODOS_PROCESSAMENTO_TOKENS = ['lemmatizacao', 'stemming', 'none']

# Stopwords extras a serem adicionadas às stopwords padrão do spaCy
STOPWORDS_EXTRAS = []
# STOPWORDS_EXTRAS = ['o', 'a', 'os', 'as', 'de', 'do', 'da', 'dos', 'das', 'em', 'no', 'na', 'nos', 'nas']
