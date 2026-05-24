import os

DIRETORIO_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAMINHO_PARQUET_ENTRADA = os.path.join(DIRETORIO_BASE, "input", "artigos_anotacao_lg.parquet")
DIRETORIO_SAIDA = os.path.join(DIRETORIO_BASE, "output")
CAMINHO_LOG = os.path.join(DIRETORIO_SAIDA, "fase2_pipeline.log")

METODOS_EMBEDDING = ["bow", "tfidf", "word2vec"]
TOP_K_RESULTADOS = 10

PARAMS_BOW = {"max_features": 5000, "min_df": 1}
PARAMS_TFIDF = {"max_features": 5000, "min_df": 1, "norm": "l2"}
PARAMS_WORD2VEC = {"vector_size": 100, "window": 5, "min_count": 1, "epochs": 30, "seed": 42}

HABILITAR_TSNE = True
PARAMS_TSNE = {"n_components": 2, "perplexity": 30, "n_iter": 2000, "random_state": 42}

PARAMS_PLOT_TSNE = {
    "figsize": (16, 12),
    "dpi": 150,
    "marker_size": 50,
    "annotate_fontsize": 7,
}

CAMINHO_SAIDA_TSNE = os.path.join(DIRETORIO_SAIDA, "tsne_plot.png")
