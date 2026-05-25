import pandas as pd
from preprocessing import obter_instancia_nlp, aplicar_stemming, normalizar_texto, extrair_ngramas
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


def _processar_ngramas(id_artigo, artigo, documento, metodo_processamento, n, tipo_tokenizacao):
    """Gera linhas com tokenização por n-grama (bigrama ou trigrama)."""
    tokens_base = list(documento)
    grupos = extrair_ngramas(tokens_base, n)
    linhas = []
    for id_token, grupo in enumerate(grupos, start=1):
        texto_grupo = " ".join(t.text for t in grupo)
        lema_grupo = " ".join(t.lemma_ for t in grupo)
        if metodo_processamento == 'lemmatizacao':
            processado = lema_grupo
        elif metodo_processamento == 'stemming':
            processado = " ".join(aplicar_stemming(t.lemma_) for t in grupo)
        else:
            processado = texto_grupo
        entidade, rotulo_entidade = _extrair_entidade(grupo[0])
        linhas.append(_construir_linha(
            id_artigo, id_token, texto_grupo, lema_grupo, processado,
            grupo[0].pos_, grupo[0].tag_, grupo[0].dep_,
            grupo[0].head.text if grupo[0].head else "",
            entidade, rotulo_entidade, tipo_tokenizacao, artigo,
        ))
    return linhas


def _processar_sentencas(id_artigo, artigo, documento, metodo_processamento):
    """Gera linhas com tokenização por sentença completa."""
    linhas = []
    for id_sentenca, sentenca in enumerate(documento.sents, start=1):
        texto_sentenca = sentenca.text
        lema_sentenca = " ".join(t.lemma_ for t in sentenca)
        if metodo_processamento == 'lemmatizacao':
            processado = lema_sentenca
        elif metodo_processamento == 'stemming':
            processado = " ".join(aplicar_stemming(t.lemma_) for t in sentenca)
        else:
            processado = texto_sentenca
        linhas.append(_construir_linha(
            id_artigo, id_sentenca, texto_sentenca, lema_sentenca, processado,
            "SENT", "SENT", "", "", "", "", 'sentenca', artigo,
        ))
    return linhas


def processar_lote_artigos(artigos, metodo_processamento='none', tipo_tokenizacao='palavra'):
    """
    Processa uma lista de artigos em lote com spaCy.
    Aplica normalização textual antes do processamento linguístico.

    Args:
        artigos: Lista de dicionários de artigos (titulo, url, conteudo).
        metodo_processamento: 'none', 'lemmatizacao' ou 'stemming'.
        tipo_tokenizacao: 'palavra', 'bigrama', 'trigrama' ou 'sentenca'.

    Returns:
        DataFrame pandas com todos os tokens anotados.
    """
    nlp = obter_instancia_nlp()
    textos = [normalizar_texto(artigo["conteudo"]) for artigo in artigos]
    logger.info(
        "Processando %d artigos em lote (batch_size=%d, metodo=%s, tipo=%s)...",
        len(textos), TAMANHO_LOTE, metodo_processamento, tipo_tokenizacao,
    )

    linhas = []

    for id_artigo, (artigo, documento) in enumerate(
        zip(artigos, nlp.pipe(textos, batch_size=TAMANHO_LOTE)), start=1
    ):
        if tipo_tokenizacao == 'palavra':
            novas_linhas = _processar_palavras(id_artigo, artigo, documento, metodo_processamento)
        elif tipo_tokenizacao == 'bigrama':
            novas_linhas = _processar_ngramas(id_artigo, artigo, documento, metodo_processamento, 2, 'bigrama')
        elif tipo_tokenizacao == 'trigrama':
            novas_linhas = _processar_ngramas(id_artigo, artigo, documento, metodo_processamento, 3, 'trigrama')
        elif tipo_tokenizacao == 'sentenca':
            novas_linhas = _processar_sentencas(id_artigo, artigo, documento, metodo_processamento)
        else:
            logger.warning("Tipo de tokenizacao desconhecido '%s', usando 'palavra'.", tipo_tokenizacao)
            novas_linhas = _processar_palavras(id_artigo, artigo, documento, metodo_processamento)

        linhas.extend(novas_linhas)
        logger.info(
            "Artigo %d processado: '%s' -> %d tokens (%s)",
            id_artigo, artigo["titulo"], len(novas_linhas), tipo_tokenizacao,
        )

    dataframe = pd.DataFrame(linhas)
    logger.info("DataFrame criado: %d linhas, %d colunas", len(dataframe), len(dataframe.columns))
    return dataframe
