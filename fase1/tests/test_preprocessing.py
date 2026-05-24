import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest
from preprocessing import (
    obter_instancia_nlp,
    obter_stopwords,
    adicionar_stopwords_personalizadas,
    aplicar_stemming,
    tokenizar_artigo,
    remover_stopwords_dos_tokens,
)


def test_obter_instancia_nlp_retorna_modelo():
    nlp = obter_instancia_nlp()
    assert nlp is not None
    assert nlp.lang == "pt"


def test_obter_stopwords_nao_vazio():
    stopwords_atuais = obter_stopwords()
    assert len(stopwords_atuais) > 0
    assert "de" in stopwords_atuais or "o" in stopwords_atuais or "a" in stopwords_atuais


def test_adicionar_stopwords_personalizadas():
    adicionar_stopwords_personalizadas(["palavrateste"])
    stopwords_atuais = obter_stopwords()
    assert "palavrateste" in stopwords_atuais


def test_aplicar_stemming_snowball():
    resultado = aplicar_stemming("processamento", metodo='snowball')
    assert isinstance(resultado, str)
    assert len(resultado) > 0
    assert len(resultado) <= len("processamento")


def test_aplicar_stemming_porter():
    resultado = aplicar_stemming("processamento", metodo='porter')
    assert isinstance(resultado, str)
    assert len(resultado) > 0


def test_tokenizar_artigo_none():
    texto = "O processamento de linguagem natural é fascinante."
    resultado = tokenizar_artigo(texto, metodo_processamento='none')
    assert "tokens" in resultado
    assert "sentencas" in resultado
    assert len(resultado["tokens"]) > 0
    chaves_token = [
        "texto", "lema", "processado", "pos", "tag", "dependencia",
        "eh_stopword", "eh_pontuacao", "eh_alfabetico"
    ]
    for chave in chaves_token:
        assert chave in resultado["tokens"][0], f"Campo '{chave}' ausente"
    primeiro_token = resultado["tokens"][0]
    assert primeiro_token["processado"] == primeiro_token["texto"]


def test_tokenizar_artigo_lemmatizacao():
    texto = "Os computadores processam dados rapidamente."
    resultado = tokenizar_artigo(texto, metodo_processamento='lemmatizacao')
    tokens = resultado["tokens"]
    for token in tokens:
        assert token["processado"] == token["lema"]


def test_tokenizar_artigo_stemming():
    texto = "O processamento de linguagem natural é fascinante."
    resultado = tokenizar_artigo(texto, metodo_processamento='stemming')
    tokens = resultado["tokens"]
    for token in tokens:
        assert isinstance(token["processado"], str)
        assert len(token["processado"]) > 0
    houve_mudanca = any(
        token["processado"] != token["lema"] for token in tokens if token["eh_alfabetico"]
    )
    assert houve_mudanca, "Stemming deveria alterar ao menos um token"


def test_tokenizar_artigo_compatibilidade_retroativa():
    """tokenizar_artigo sem argumento deve funcionar como antes (metodo_processamento='none')."""
    texto = "O processamento de linguagem natural é fascinante."
    resultado = tokenizar_artigo(texto)
    assert "tokens" in resultado
    assert "sentencas" in resultado


def test_remover_stopwords_dos_tokens():
    texto = "O processamento de linguagem natural é fascinante."
    resultado = tokenizar_artigo(texto, metodo_processamento='lemmatizacao')
    tokens_filtrados = remover_stopwords_dos_tokens(resultado["tokens"])
    processados = [token["processado"].lower() for token in tokens_filtrados]
    assert "o" not in processados
    assert "de" not in processados
    assert "processamento" in processados or "processar" in processados

