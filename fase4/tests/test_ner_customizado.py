"""
Testes para ner_customizado.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd


def _spacy_disponivel():
    try:
        import spacy
        return True
    except ImportError:
        return False


DADOS_TREINO_SIMPLES = [
    ("A Petrobras anunciou lucro.", {"entities": [(2, 11, "ORG")]}),
    ("O Google comprou a empresa.", {"entities": [(3, 9, "ORG")]}),
    ("Lula visitou Brasilia.", {"entities": [(0, 4, "PER"), (13, 21, "LOC")]}),
]


@pytest.mark.skipif(not _spacy_disponivel(), reason="spaCy nao disponivel")
def test_treinar_ner_customizado_retorna_modelo():
    from ner_customizado import treinar_ner_customizado
    modelo = treinar_ner_customizado(DADOS_TREINO_SIMPLES, epochs=2)
    assert modelo is not None


@pytest.mark.skipif(not _spacy_disponivel(), reason="spaCy nao disponivel")
def test_treinar_ner_customizado_dados_vazios():
    from ner_customizado import treinar_ner_customizado
    modelo = treinar_ner_customizado([])
    assert modelo is not None


@pytest.mark.skipif(not _spacy_disponivel(), reason="spaCy nao disponivel")
def test_avaliar_modelo_ner_retorna_metricas():
    from ner_customizado import avaliar_modelo_ner, treinar_ner_customizado
    modelo = treinar_ner_customizado(DADOS_TREINO_SIMPLES, epochs=2)
    metricas = avaliar_modelo_ner(modelo, DADOS_TREINO_SIMPLES[:2])
    assert "precision" in metricas
    assert "recall" in metricas
    assert "f1" in metricas
    for v in metricas.values():
        assert 0.0 <= v <= 1.0


@pytest.mark.skipif(not _spacy_disponivel(), reason="spaCy nao disponivel")
def test_avaliar_modelo_ner_dados_vazios():
    from ner_customizado import avaliar_modelo_ner, treinar_ner_customizado
    modelo = treinar_ner_customizado(DADOS_TREINO_SIMPLES, epochs=1)
    metricas = avaliar_modelo_ner(modelo, [])
    assert metricas["precision"] == 0.0
    assert metricas["recall"] == 0.0
    assert metricas["f1"] == 0.0


@pytest.mark.skipif(not _spacy_disponivel(), reason="spaCy nao disponivel")
def test_comparar_modelos_retorna_dataframe():
    from ner_customizado import comparar_modelos, treinar_ner_customizado
    modelo1 = treinar_ner_customizado(DADOS_TREINO_SIMPLES, epochs=1)
    modelo2 = treinar_ner_customizado(DADOS_TREINO_SIMPLES, epochs=1)
    textos = ["A Petrobras anunciou investimentos.", "O Google comprou empresa."]
    df = comparar_modelos(modelo1, modelo2, textos)
    assert isinstance(df, pd.DataFrame)


def test_converter_entidades_formato_treino_sem_coluna_id():
    from ner_customizado import converter_entidades_para_formato_treino
    df = pd.DataFrame({
        "token": ["Petrobras", "anunciou", "lucro"],
        "entidade": ["Petrobras", None, None],
        "rotulo_entidade": ["ORG", None, None],
    })
    documentos = ["Petrobras anunciou lucro."]
    resultado = converter_entidades_para_formato_treino(df, documentos)
    assert isinstance(resultado, list)


def test_exportar_metrica_ner(tmp_path):
    from ner_customizado import exportar_metrica_ner
    metricas = {"precision": 0.8, "recall": 0.75, "f1": 0.77}
    caminho = str(tmp_path / "metricas.json")
    exportar_metrica_ner(metricas, caminho)
    assert os.path.exists(caminho)
    import json
    with open(caminho) as fh:
        dados = json.load(fh)
    assert dados["precision"] == 0.8
