import sys
import os

DIRETORIO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIRETORIO_SCRIPT)

from fase2_config import (
    CAMINHO_PARQUET_ENTRADA,
    DIRETORIO_SAIDA,
    METODOS_EMBEDDING,
    TOP_K_RESULTADOS,
    PARAMS_BOW,
    PARAMS_TFIDF,
    PARAMS_WORD2VEC,
    HABILITAR_TSNE,
    PARAMS_TSNE,
    PARAMS_PLOT_TSNE,
    CAMINHO_SAIDA_TSNE,
)
from logger import inicializar_sistema_log
from embedding_pipeline import PipelineEmbeddings
from search_interface import iniciar_interface_busca

os.makedirs(DIRETORIO_SAIDA, exist_ok=True)

logger = inicializar_sistema_log("fase2")


def executar_fase2_principal():
    logger.info("=" * 60)
    logger.info("FASE 2: Token Embedding e Busca por Similaridade")
    logger.info("=" * 60)

    configuracoes = {
        "METODOS_EMBEDDING": METODOS_EMBEDDING,
        "TOP_K_RESULTADOS": TOP_K_RESULTADOS,
        "PARAMS_BOW": PARAMS_BOW,
        "PARAMS_TFIDF": PARAMS_TFIDF,
        "PARAMS_WORD2VEC": PARAMS_WORD2VEC,
        "HABILITAR_TSNE": HABILITAR_TSNE,
        "PARAMS_TSNE": PARAMS_TSNE,
        "PARAMS_PLOT_TSNE": PARAMS_PLOT_TSNE,
        "CAMINHO_SAIDA_TSNE": CAMINHO_SAIDA_TSNE,
    }

    pipeline = PipelineEmbeddings(configuracoes, CAMINHO_PARQUET_ENTRADA)
    motores_busca = pipeline.executar()

    if not motores_busca:
        logger.error("Nenhum search engine foi treinado. Verifique a configuracao.")
        return

    iniciar_interface_busca(pipeline)

    logger.info("=" * 60)
    logger.info("FASE 2 CONCLUIDA")
    logger.info("=" * 60)


if __name__ == "__main__":
    executar_fase2_principal()
