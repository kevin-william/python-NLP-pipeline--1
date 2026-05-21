import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from fase2_config import (
    INPUT_CSV,
    OUTPUT_DIR,
    EMBEDDING_METHODS,
    TOP_K_RESULTS,
    BOW_PARAMS,
    TFIDF_PARAMS,
    WORD2VEC_PARAMS,
    ENABLE_TSNE,
    TSNE_PARAMS,
    TSNE_PLOT_PARAMS,
    TSNE_OUTPUT,
)
from logger import setup_logger
from embedding_pipeline import EmbeddingPipeline
from search_interface import start_search_interface

os.makedirs(OUTPUT_DIR, exist_ok=True)

logger = setup_logger("fase2")


def main():
    logger.info("=" * 60)
    logger.info("FASE 2: Token Embedding e Busca por Similaridade")
    logger.info("=" * 60)

    config = {
        "EMBEDDING_METHODS": EMBEDDING_METHODS,
        "TOP_K_RESULTS": TOP_K_RESULTS,
        "BOW_PARAMS": BOW_PARAMS,
        "TFIDF_PARAMS": TFIDF_PARAMS,
        "WORD2VEC_PARAMS": WORD2VEC_PARAMS,
        "ENABLE_TSNE": ENABLE_TSNE,
        "TSNE_PARAMS": TSNE_PARAMS,
        "TSNE_PLOT_PARAMS": TSNE_PLOT_PARAMS,
        "TSNE_OUTPUT": TSNE_OUTPUT,
    }

    pipeline = EmbeddingPipeline(config, INPUT_CSV)
    search_engines = pipeline.run()

    if not search_engines:
        logger.error("Nenhum search engine foi treinado. Verifique a configuracao.")
        return

    start_search_interface(pipeline)

    logger.info("=" * 60)
    logger.info("FASE 2 CONCLUIDA")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
