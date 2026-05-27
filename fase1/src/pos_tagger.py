import pandas as pd
from preprocessing import obter_instancia_nlp, aplicar_stemming, normalizar_texto
from fase1_config import TAMANHO_LOTE
from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)


def _aplicar_metodo(lema, texto, metodo_processamento):
    """Retorna o valor processado de um token conforme o método escolhido."""
    if metodo_processamento == 'lemmatizacao':
        return lema
    if metodo_processamento == 'stemming':
        return aplicar_stemming(lema)
    return texto


def _extrair_entidade(token):
    """Retorna (entidade, rotulo_entidade) para um token spaCy."""
    if token.ent_iob_ in ("B", "I"):
        return token.ent_type_ or "", token.ent_type_ or ""
    return "", ""


def _construir_linha(id_artigo, id_token, texto, lema, processado, pos, tag,
                     dep, cabeca, entidade, rotulo_entidade, tipo_tokenizacao, artigo):
    """Monta o dicionário de uma linha do DataFrame final."""
    return {
        "id_artigo": id_artigo,
        "id_token": id_token,
        "token": texto,
        "pos": pos,
        "tag": tag,
        "lema": lema,
        "processado": processado,
        "relacao_dependencia": dep,
        "token_cabeca": cabeca,
        "entidade": entidade,
        "rotulo_entidade": rotulo_entidade,
        "tipo_tokenizacao": tipo_tokenizacao,
        "titulo": artigo["titulo"],
        "url": artigo["url"],
    }


def _processar_palavras(id_artigo, artigo, documento, metodo_processamento):
    """Gera linhas com tokenização por palavra (unigrama)."""
    linhas = []
    for id_token, token in enumerate(documento, start=1):
        lema = token.lemma_
        processado = _aplicar_metodo(lema, token.text, metodo_processamento)
        entidade, rotulo_entidade = _extrair_entidade(token)
        linhas.append(_construir_linha(
            id_artigo, id_token, token.text, lema, processado,
            token.pos_, token.tag_, token.dep_,
            token.head.text if token.head else "",
            entidade, rotulo_entidade, 'palavra', artigo,
        ))
    return linhas


def processar_lote_artigos(artigos, metodo_processamento='none'):
    """
    Processa uma lista de artigos em lote com spaCy, tokenizando por palavra.
    Aplica normalização textual antes do processamento linguístico.

    Args:
        artigos: Lista de dicionários de artigos (titulo, url, conteudo).
        metodo_processamento: 'none', 'lemmatizacao' ou 'stemming'.

    Returns:
        DataFrame pandas com todos os tokens anotados.
    """
    nlp = obter_instancia_nlp()
    textos = [normalizar_texto(artigo["conteudo"]) for artigo in artigos]
    logger.info(
        "Processando %d artigos em lote (batch_size=%d, metodo=%s)...",
        len(textos), TAMANHO_LOTE, metodo_processamento,
    )

    linhas = []
    for id_artigo, (artigo, documento) in enumerate(
        zip(artigos, nlp.pipe(textos, batch_size=TAMANHO_LOTE)), start=1
    ):
        novas_linhas = _processar_palavras(id_artigo, artigo, documento, metodo_processamento)
        linhas.extend(novas_linhas)
        logger.info(
            "Artigo %d processado: '%s' -> %d tokens",
            id_artigo, artigo["titulo"], len(novas_linhas),
        )

    dataframe = pd.DataFrame(linhas)
    logger.info("DataFrame criado: %d linhas, %d colunas", len(dataframe), len(dataframe.columns))
    return dataframe
