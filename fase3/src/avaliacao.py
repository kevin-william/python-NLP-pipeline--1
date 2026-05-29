import numpy as np
import pandas as pd

from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)


def _indice_de_qual_palavra(palavra, documentos_tokenizados):
    return [i for i, tokens in enumerate(documentos_tokenizados) if palavra in tokens]


def calcular_coerencia_top_n(topicos, documentos, top_n=None):
    if not topicos or not documentos:
        return 0.0

    documentos_tokenizados = [set(doc.split()) for doc in documentos]

    coerencias = []
    for topico in topicos:
        palavras = topico[:top_n] if top_n else topico
        if len(palavras) < 2:
            coerencias.append(0.0)
            continue

        pares_scores = []
        for i in range(len(palavras)):
            docs_i = set(_indice_de_qual_palavra(palavras[i], documentos_tokenizados))
            for j in range(i + 1, len(palavras)):
                docs_j = set(_indice_de_qual_palavra(palavras[j], documentos_tokenizados))
                intersecao = len(docs_i & docs_j)
                uniao = len(docs_i | docs_j)
                if uniao > 0:
                    pares_scores.append(intersecao / uniao)
                else:
                    pares_scores.append(0.0)

        if pares_scores:
            coerencias.append(float(np.mean(pares_scores)))
        else:
            coerencias.append(0.0)

    return float(np.mean(coerencias)) if coerencias else 0.0


def comparar_modelos(resultados):
    linhas = []
    for nome_modelo, dados in resultados.items():
        linhas.append({
            "modelo": nome_modelo,
            "num_topicos": dados.get("num_topicos", 0),
            "coerencia_media": dados.get("coerencia", 0.0),
            "perplexidade": dados.get("perplexidade", None),
        })

    dataframe = pd.DataFrame(linhas)
    logger.info("Tabela comparativa gerada com %d modelos", len(dataframe))
    return dataframe
