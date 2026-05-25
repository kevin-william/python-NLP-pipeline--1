import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest
from corpus_loader import carregar_artigos, filtrar_artigos_por_tamanho
from preprocessing import (
    tokenizar_artigo,
    tokenizar_por_tipo,
    remover_stopwords_dos_tokens,
    obter_stopwords,
)
from vocab_analysis import analisar_vocabulario
from collections import Counter


def _tokenizar_e_filtrar(conteudo, metodo='lemmatizacao'):
    resultado = tokenizar_artigo(conteudo, metodo_processamento=metodo)
    tokens = resultado["tokens"]
    tokens_filtrados = remover_stopwords_dos_tokens(tokens)
    return tokens, tokens_filtrados


def test_pipeline_minima_completa():
    artigos = carregar_artigos()
    tokens, tokens_filtrados = _tokenizar_e_filtrar(artigos[0]["conteudo"])

    contador_lema_bruto = Counter(token["processado"].lower() for token in tokens if token["processado"].strip())
    contador_lema_filtrado = Counter(token["processado"].lower() for token in tokens_filtrados if token["processado"].strip())

    assert len(contador_lema_bruto) > 0
    assert len(contador_lema_filtrado) > 0
    assert len(contador_lema_filtrado) <= len(contador_lema_bruto)


def test_pipeline_minima_stemming():
    artigos = carregar_artigos()
    tokens, tokens_filtrados = _tokenizar_e_filtrar(artigos[0]["conteudo"], metodo='stemming')

    contador_bruto = Counter(token["processado"].lower() for token in tokens if token["processado"].strip())
    contador_filtrado = Counter(token["processado"].lower() for token in tokens_filtrados if token["processado"].strip())

    assert len(contador_bruto) > 0
    assert len(contador_filtrado) <= len(contador_bruto)


def test_efeito_remocao_stopwords():
    artigos = carregar_artigos()
    conteudo = artigos[0]["conteudo"]
    tokens, tokens_filtrados = _tokenizar_e_filtrar(conteudo)
    stopwords_atuais = obter_stopwords()

    stopwords_apos_filtro = sum(
        1 for token in tokens_filtrados if token["processado"].lower() in stopwords_atuais
    )
    assert stopwords_apos_filtro == 0, "Nao deveria haver stopwords apos remocao"
    assert len(tokens_filtrados) <= len(tokens), "Filtro deveria reduzir ou manter tokens"


def test_saida_analise_vocabulario():
    artigos = carregar_artigos()
    conteudo = artigos[0]["conteudo"]
    resultado = tokenizar_artigo(conteudo, metodo_processamento='lemmatizacao')
    tokens = resultado["tokens"]

    tokens_info = [
        {"texto": token["texto"], "lema": token["processado"], "pos": token["pos"]}
        for token in tokens
    ]
    tokens_filtrados = remover_stopwords_dos_tokens(tokens)
    tokens_filtrados_info = [
        {"texto": token["texto"], "lema": token["processado"], "pos": token["pos"]}
        for token in tokens_filtrados
    ]

    analise = analisar_vocabulario(tokens_info, tokens_filtrados_info)
    assert "quantidade_vocabulario_bruto" in analise
    assert "quantidade_vocabulario_filtrado" in analise
    assert "percentual_reducao_vocabulario" in analise
    assert analise["quantidade_vocabulario_filtrado"] <= analise["quantidade_vocabulario_bruto"]


# ---------------------------------------------------------------------------
# Testes de pipeline com novos tipos de tokenização
# ---------------------------------------------------------------------------

def test_pipeline_com_tipo_bigrama():
    artigos = carregar_artigos()
    resultado = tokenizar_por_tipo(artigos[0]["conteudo"], tipo_tokenizacao='bigrama')
    tokens = resultado["tokens"]
    assert len(tokens) > 0
    assert " " in tokens[0]["texto"], "Primeiro token deve ser um bigrama (com espaço)"


def test_pipeline_com_tipo_trigrama():
    artigos = carregar_artigos()
    resultado = tokenizar_por_tipo(artigos[0]["conteudo"], tipo_tokenizacao='trigrama')
    tokens = resultado["tokens"]
    assert len(tokens) > 0
    partes = tokens[0]["texto"].split(" ")
    assert len(partes) == 3, "Primeiro token deve ser um trigrama (3 termos)"


def test_pipeline_com_tipo_sentenca():
    artigos = carregar_artigos()
    resultado = tokenizar_por_tipo(artigos[0]["conteudo"], tipo_tokenizacao='sentenca')
    tokens = resultado["tokens"]
    assert len(tokens) > 0
    for token in tokens:
        assert token["pos"] == "SENT"


# ---------------------------------------------------------------------------
# Testes de filtro mínimo de palavras na pipeline
# ---------------------------------------------------------------------------

def test_filtro_minimo_nao_remove_artigos_com_limiar_zero():
    artigos = carregar_artigos()
    validos, removidos = filtrar_artigos_por_tamanho(artigos, minimo_palavras=0)
    assert len(validos) == len(artigos)
    assert len(removidos) == 0


def test_filtro_minimo_remove_artigos_com_limiar_alto():
    artigos = carregar_artigos()
    validos, removidos = filtrar_artigos_por_tamanho(artigos, minimo_palavras=100000)
    assert len(removidos) > 0
    assert len(validos) + len(removidos) == len(artigos)

