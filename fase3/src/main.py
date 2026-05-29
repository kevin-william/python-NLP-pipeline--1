import os
import sys

DIRETORIO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIRETORIO_SCRIPT)
sys.path.insert(0, os.path.join(DIRETORIO_SCRIPT, "..", "..", "shared"))
sys.path.insert(0, os.path.join(DIRETORIO_SCRIPT, "modelos_topicos"))

from fase3_config import (
    CAMINHO_ARTEFATO_FASE2,
    DIRETORIO_PLOTS,
    DIRETORIO_SAIDA,
    NUM_TOPICOS,
    PARAMS_LSA,
    PARAMS_LDA,
    PARAMS_NMF,
    PARAMS_BOW_RECONSTRUCAO,
    TOP_N_PALAVRAS,
)
from logger import inicializar_sistema_log
from carregador_artefato import carregar_artefato_fase2
from eda import executar_eda
from modelos_topicos import ModeloLSA, ModeloLDA, ModeloNMF
from avaliacao import calcular_coerencia_top_n, comparar_modelos
from visualizacao import (
    plotar_top_palavras_por_topico,
    plotar_distribuicao_topicos_documentos,
    plotar_comparacao_metricas,
)

os.makedirs(DIRETORIO_SAIDA, exist_ok=True)
os.makedirs(DIRETORIO_PLOTS, exist_ok=True)

logger = inicializar_sistema_log("fase3")


def executar_fase3_principal():
    logger.info("=" * 60)
    logger.info("FASE 3: Modelagem de Topicos (LSA, LDA, NMF)")
    logger.info("=" * 60)

    # 1. Carregar artefato
    logger.info("--- Etapa 1: Carregando artefato ---")
    artefato = carregar_artefato_fase2(CAMINHO_ARTEFATO_FASE2)

    # 2. EDA
    logger.info("--- Etapa 2: Analise Exploratoria ---")
    metricas_eda = executar_eda(artefato.documentos, artefato.titulos, DIRETORIO_PLOTS)

    # 3. Obter vocabulario da matriz TF-IDF
    vocabulario = artefato.tfidf_vectorizer.get_feature_names_out().tolist()

    # 4. Treinar modelos
    logger.info("--- Etapa 3: Treinando modelos de topicos ---")

    # LSA
    lsa = ModeloLSA(NUM_TOPICOS, PARAMS_LSA)
    lsa.treinar(artefato.tfidf_matrix)
    topicos_lsa = lsa.obter_topicos(vocabulario, TOP_N_PALAVRAS)
    topicos_lsa_com_pesos = lsa.obter_topicos_com_pesos(vocabulario, TOP_N_PALAVRAS)
    logger.info("Topicos LSA: %s", topicos_lsa)

    # LDA (usa BoW matrix; reconstrói se necessario)
    lda = ModeloLDA(NUM_TOPICOS, PARAMS_LDA)
    matriz_bow = artefato.bow_matrix
    if matriz_bow is None:
        logger.info("Reconstruindo matriz BoW para LDA via CountVectorizer")
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(**PARAMS_BOW_RECONSTRUCAO)
        matriz_bow = cv.fit_transform(artefato.documentos)
        vocabulario_bow = cv.get_feature_names_out().tolist()
        lda.treinar(matriz_bow)
        topicos_lda = lda.obter_topicos(vocabulario_bow, TOP_N_PALAVRAS)
        topicos_lda_com_pesos = lda.obter_topicos_com_pesos(vocabulario_bow, TOP_N_PALAVRAS)
        perplexidade = lda.obter_perplexidade(matriz_bow)
    else:
        vocabulario_bow = artefato.bow_vectorizer.get_feature_names_out().tolist()
        lda.treinar(matriz_bow)
        topicos_lda = lda.obter_topicos(vocabulario_bow, TOP_N_PALAVRAS)
        topicos_lda_com_pesos = lda.obter_topicos_com_pesos(vocabulario_bow, TOP_N_PALAVRAS)
        perplexidade = lda.obter_perplexidade(matriz_bow)
    logger.info("Topicos LDA: %s", topicos_lda)
    logger.info("Perplexidade LDA: %.4f", perplexidade)

    # NMF
    nmf = ModeloNMF(NUM_TOPICOS, PARAMS_NMF)
    nmf.treinar(artefato.tfidf_matrix)
    topicos_nmf = nmf.obter_topicos(vocabulario, TOP_N_PALAVRAS)
    topicos_nmf_com_pesos = nmf.obter_topicos_com_pesos(vocabulario, TOP_N_PALAVRAS)
    logger.info("Topicos NMF: %s", topicos_nmf)

    # 5. Avaliacao
    logger.info("--- Etapa 4: Avaliando modelos ---")
    coerencia_lsa = calcular_coerencia_top_n(topicos_lsa, artefato.documentos)
    coerencia_lda = calcular_coerencia_top_n(topicos_lda, artefato.documentos)
    coerencia_nmf = calcular_coerencia_top_n(topicos_nmf, artefato.documentos)

    logger.info("Coerencia LSA: %.4f", coerencia_lsa)
    logger.info("Coerencia LDA: %.4f", coerencia_lda)
    logger.info("Coerencia NMF: %.4f", coerencia_nmf)

    resultados = {
        "LSA": {"num_topicos": NUM_TOPICOS, "coerencia": coerencia_lsa, "perplexidade": None},
        "LDA": {"num_topicos": NUM_TOPICOS, "coerencia": coerencia_lda, "perplexidade": perplexidade},
        "NMF": {"num_topicos": NUM_TOPICOS, "coerencia": coerencia_nmf, "perplexidade": None},
    }

    dataframe = comparar_modelos(resultados)
    caminho_csv = os.path.join(DIRETORIO_SAIDA, "comparacao_modelos.csv")
    dataframe.to_csv(caminho_csv, index=False, encoding="utf-8")
    logger.info("Tabela comparativa salva em: %s", caminho_csv)

    # 6. Visualizacoes
    logger.info("--- Etapa 5: Gerando visualizacoes ---")

    plotar_top_palavras_por_topico(topicos_lsa_com_pesos, "lsa", DIRETORIO_PLOTS)
    plotar_top_palavras_por_topico(topicos_lda_com_pesos, "lda", DIRETORIO_PLOTS)
    plotar_top_palavras_por_topico(topicos_nmf_com_pesos, "nmf", DIRETORIO_PLOTS)

    plotar_distribuicao_topicos_documentos(lsa.obter_distribuicao_documentos(),
                                           artefato.titulos, "lsa", DIRETORIO_PLOTS)
    plotar_distribuicao_topicos_documentos(lda.obter_distribuicao_documentos(),
                                           artefato.titulos, "lda", DIRETORIO_PLOTS)
    plotar_distribuicao_topicos_documentos(nmf.obter_distribuicao_documentos(),
                                           artefato.titulos, "nmf", DIRETORIO_PLOTS)

    plotar_comparacao_metricas(dataframe, DIRETORIO_PLOTS)

    # Sumario final
    logger.info("=" * 60)
    logger.info("SUMARIO FINAL DA FASE 3")
    logger.info("=" * 60)
    logger.info("Documentos analisados: %d", len(artefato.documentos))
    logger.info("Tamanho do vocabulario: %d", metricas_eda["tamanho_vocabulario"])
    logger.info("Numero de topicos: %d", NUM_TOPICOS)
    logger.info("\n%s", dataframe.to_string(index=False))
    logger.info("=" * 60)
    logger.info("FASE 3 CONCLUIDA")
    logger.info("=" * 60)

    return {
        "metricas_eda": metricas_eda,
        "dataframe_comparacao": dataframe,
        "topicos_lsa": topicos_lsa,
        "topicos_lda": topicos_lda,
        "topicos_nmf": topicos_nmf,
    }


if __name__ == "__main__":
    executar_fase3_principal()
