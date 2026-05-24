import pandas as pd
from preprocessing import obter_instancia_nlp, aplicar_stemming
from fase1_config import TAMANHO_LOTE
from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)


def processar_lote_artigos(artigos, metodo_processamento='none'):
    nlp = obter_instancia_nlp()
    textos = [artigo["conteudo"] for artigo in artigos]
    logger.info(
        "Processando %d artigos em lote (batch_size=%d, method=%s)...",
        len(textos), TAMANHO_LOTE, metodo_processamento,
    )

    linhas = []

    for id_artigo, (artigo, documento) in enumerate(
        zip(artigos, nlp.pipe(textos, batch_size=TAMANHO_LOTE)), start=1
    ):
        for id_token, token in enumerate(documento, start=1):
            lema = token.lemma_
            if metodo_processamento == 'lemmatizacao':
                processado = lema
            elif metodo_processamento == 'stemming':
                processado = aplicar_stemming(lema)
            else:
                processado = token.text

            entidade = ""
            rotulo_entidade = ""
            if token.ent_iob_ == "B" or token.ent_iob_ == "I":
                entidade = token.ent_type_ or ""
                rotulo_entidade = token.ent_type_ or ""
            linhas.append(
                {
                    "id_artigo": id_artigo,
                    "id_token": id_token,
                    "token": token.text,
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "lema": lema,
                    "processado": processado,
                    "relacao_dependencia": token.dep_,
                    "token_cabeca": token.head.text if token.head else "",
                    "entidade": entidade,
                    "rotulo_entidade": rotulo_entidade,
                    "titulo": artigo["titulo"],
                    "url": artigo["url"],
                }
            )
        logger.info(
            "Artigo %d processado: '%s' -> %d tokens",
            id_artigo,
            artigo["titulo"],
            id_token,
        )

    dataframe = pd.DataFrame(linhas)
    logger.info("DataFrame criado: %d linhas, %d colunas", len(dataframe), len(dataframe.columns))
    return dataframe
