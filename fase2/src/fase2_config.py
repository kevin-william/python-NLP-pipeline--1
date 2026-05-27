import os

# Diretorio raiz da fase2 (usado para construir caminhos relativos do projeto).
DIRETORIO_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Caminho do parquet de entrada gerado pela fase1 para o recorte atual.
CAMINHO_PARQUET_ENTRADA = os.path.join(DIRETORIO_BASE, "input", "100-artigos_anotacao_lg_lemmatizacao_palavra.parquet")

# Pasta padrao para salvar logs e artefatos da fase2.
DIRETORIO_SAIDA = os.path.join(DIRETORIO_BASE, "output")

# Caminho do arquivo de log principal da fase2.
CAMINHO_LOG = os.path.join(DIRETORIO_SAIDA, "100-artigos_anotacao_lg_none_palavra.log")

# Ordem dos metodos de embedding a treinar e disponibilizar na busca.
METODOS_EMBEDDING = ["bow", "tfidf", "word2vec"]

# Quantidade padrao de resultados retornados por consulta textual.
TOP_K_RESULTADOS = 5

# Parametros do CountVectorizer (Bag-of-Words).
PARAMS_BOW = {"max_features": 5000, "min_df": 1}

# Parametros do TfidfVectorizer.
PARAMS_TFIDF = {"max_features": 5000, "min_df": 1, "norm": "l2"}

# Parametros de treino do Word2Vec (Gensim).
PARAMS_WORD2VEC = {"vector_size": 100, "window": 5, "min_count": 1, "epochs": 30, "seed": 42}

# Habilita/desabilita a geracao do grafico t-SNE apos o treino.
HABILITAR_TSNE = True

# Parametros de reducao de dimensionalidade do t-SNE.
PARAMS_TSNE = {"n_components": 2, "perplexity": 30, "n_iter": 2000, "random_state": 42}

# Parametros visuais do plot t-SNE salvo em arquivo.
PARAMS_PLOT_TSNE = {
    "figsize": (20, 16),
    "dpi": 600,
    "marker_size": 50,
    "annotate_fontsize": 7,
}

# Caminho final da imagem t-SNE gerada no pipeline.
CAMINHO_SAIDA_TSNE = os.path.join(DIRETORIO_SAIDA, "100-artigos_anotacao_lg_none_palavra.png")
