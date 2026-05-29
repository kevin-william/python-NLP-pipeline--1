import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_TESTS_DIR, "..", "src")
_MODELOS_DIR = os.path.join(_SRC_DIR, "modelos_topicos")
_SHARED_DIR = os.path.join(_TESTS_DIR, "..", "..", "shared")
sys.path.insert(0, _SRC_DIR)
sys.path.insert(0, _MODELOS_DIR)
sys.path.insert(0, _SHARED_DIR)

import pytest


class TestAvaliacao:

    @pytest.fixture
    def topicos_teste(self):
        return [
            ["machine", "learning", "algorithm"],
            ["deep", "neural", "network"],
            ["natural", "language", "processing"],
        ]

    @pytest.fixture
    def documentos_teste(self):
        return [
            "machine learning algorithm data science",
            "deep learning neural network model",
            "natural language processing text analysis",
            "machine neural language system",
            "deep learning algorithm optimization",
        ]

    def test_coerencia_entre_0_e_1(self, topicos_teste, documentos_teste):
        from avaliacao import calcular_coerencia_top_n

        coef = calcular_coerencia_top_n(topicos_teste, documentos_teste)
        assert 0.0 <= coef <= 1.0

    def test_coerencia_documentos_vazios_retorna_zero(self):
        from avaliacao import calcular_coerencia_top_n

        coef = calcular_coerencia_top_n([["a", "b"]], [])
        assert coef == 0.0

    def test_coerencia_topicos_vazios_retorna_zero(self):
        from avaliacao import calcular_coerencia_top_n

        coef = calcular_coerencia_top_n([], ["doc1", "doc2"])
        assert coef == 0.0

    def test_comparar_modelos_retorna_dataframe_com_3_linhas(self, topicos_teste, documentos_teste):
        from avaliacao import calcular_coerencia_top_n, comparar_modelos

        resultados = {
            "LSA": {
                "num_topicos": 5,
                "coerencia": calcular_coerencia_top_n(topicos_teste, documentos_teste),
                "perplexidade": None,
            },
            "LDA": {
                "num_topicos": 5,
                "coerencia": calcular_coerencia_top_n(topicos_teste, documentos_teste),
                "perplexidade": 1234.5,
            },
            "NMF": {
                "num_topicos": 5,
                "coerencia": calcular_coerencia_top_n(topicos_teste, documentos_teste),
                "perplexidade": None,
            },
        }

        df = comparar_modelos(resultados)
        assert len(df) == 3
        assert list(df.columns) == ["modelo", "num_topicos", "coerencia_media", "perplexidade"]
        assert set(df["modelo"].tolist()) == {"LSA", "LDA", "NMF"}
