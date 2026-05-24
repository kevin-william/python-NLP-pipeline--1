import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest
from pos_tagger import processar_lote_artigos

_ARTIGOS = [
    {
        "titulo": "Teste",
        "url": "https://example.com",
        "conteudo": "O processamento de linguagem natural é uma área fascinante da inteligência artificial.",
    }
]


def test_processar_lote_artigos():
    dataframe = processar_lote_artigos(_ARTIGOS)
    assert len(dataframe) > 0
    colunas_esperadas = [
        "id_artigo", "id_token", "token", "pos", "tag",
        "lema", "processado", "relacao_dependencia", "token_cabeca", "entidade", "rotulo_entidade",
        "titulo", "url",
    ]
    for coluna in colunas_esperadas:
        assert coluna in dataframe.columns, f"Coluna '{coluna}' ausente"
    assert dataframe["id_artigo"].iloc[0] == 1
    assert dataframe["titulo"].iloc[0] == "Teste"
    assert dataframe["url"].iloc[0] == "https://example.com"


def test_coluna_processado_none():
    dataframe = processar_lote_artigos(_ARTIGOS, metodo_processamento='none')
    assert (dataframe["processado"] == dataframe["token"]).all(), \
        "Com method='none', 'processed' deve ser igual ao token original"


def test_coluna_processado_lemmatizacao():
    dataframe = processar_lote_artigos(_ARTIGOS, metodo_processamento='lemmatizacao')
    assert (dataframe["processado"] == dataframe["lema"]).all(), \
        "Com method='lemmatizacao', 'processed' deve ser igual ao lemma"


def test_coluna_processado_stemming():
    dataframe = processar_lote_artigos(_ARTIGOS, metodo_processamento='stemming')
    dataframe_alfabetico = dataframe[dataframe["token"].str.isalpha()]
    houve_mudanca = (dataframe_alfabetico["processado"] != dataframe_alfabetico["lema"]).any()
    assert houve_mudanca, "Stemming deveria alterar ao menos um token"


def test_pos_tags_nao_vazio():
    artigos = [
        {
            "titulo": "Teste 2",
            "url": "https://example.com/2",
            "conteudo": "O gato preto correu rapidamente.",
        }
    ]
    dataframe = processar_lote_artigos(artigos)
    valores_pos = dataframe["pos"].unique()
    assert len(valores_pos) > 0, "Deveria ter POS tags"
    assert "VERB" in valores_pos or "NOUN" in valores_pos


def test_lema_presente():
    artigos = [
        {
            "titulo": "Teste 3",
            "url": "https://example.com/3",
            "conteudo": "Os computadores processam dados.",
        }
    ]
    dataframe = processar_lote_artigos(artigos)
    lemas = dataframe["lema"].dropna().tolist()
    assert len(lemas) > 0
    lemas_minusculos = [lema.lower() for lema in lemas]
    assert any("computador" in lema for lema in lemas_minusculos)

