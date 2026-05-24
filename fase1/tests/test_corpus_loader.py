import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest
from corpus_loader import carregar_artigos, obter_estatisticas_corpus


def test_carregar_artigos_contagem():
    artigos = carregar_artigos()
    assert len(artigos) > 0, "Deveria carregar ao menos 1 artigo"


def test_estrutura_artigo():
    artigos = carregar_artigos()
    for artigo in artigos:
        assert "titulo" in artigo
        assert "url" in artigo
        assert "conteudo" in artigo
        assert isinstance(artigo["titulo"], str) and len(artigo["titulo"]) > 0
        assert isinstance(artigo["url"], str) and len(artigo["url"]) > 0
        assert isinstance(artigo["conteudo"], str)


def test_titulos_artigos():
    artigos = carregar_artigos()
    titulos = [artigo["titulo"] for artigo in artigos]
    assert len(titulos) > 0
    assert any("Python" in titulo for titulo in titulos)


def test_obter_estatisticas_corpus():
    artigos = carregar_artigos()
    estatisticas = obter_estatisticas_corpus(artigos)
    assert estatisticas["total_artigos"] == len(artigos)
    assert estatisticas["total_caracteres"] > 0
    assert estatisticas["media_caracteres_por_artigo"] > 0
