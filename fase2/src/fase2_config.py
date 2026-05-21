import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_CSV = os.path.join(BASE_DIR, "input", "artigos_anotacao_lg.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_FILE = os.path.join(OUTPUT_DIR, "fase2_pipeline.log")

EMBEDDING_METHODS = ["bow", "tfidf", "word2vec"]
TOP_K_RESULTS = 10

BOW_PARAMS = {"max_features": 5000, "min_df": 1}
TFIDF_PARAMS = {"max_features": 5000, "min_df": 1, "norm": "l2"}
WORD2VEC_PARAMS = {"vector_size": 100, "window": 5, "min_count": 1, "epochs": 30, "seed": 42}

ENABLE_TSNE = True
TSNE_PARAMS = {"n_components": 2, "perplexity": 30, "n_iter": 2000, "random_state": 42}

TSNE_PLOT_PARAMS = {
    "figsize": (16, 12),
    "dpi": 150,
    "marker_size": 50,
    "annotate_fontsize": 7,
}

TSNE_OUTPUT = os.path.join(OUTPUT_DIR, "tsne_plot.png")
