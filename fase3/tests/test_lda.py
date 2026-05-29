import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_TESTS_DIR, "..", "src")
_MODELOS_DIR = os.path.join(_SRC_DIR, "modelos_topicos")
_SHARED_DIR = os.path.join(_TESTS_DIR, "..", "..", "shared")
sys.path.insert(0, _SRC_DIR)
sys.path.insert(0, _MODELOS_DIR)
sys.path.insert(0, _SHARED_DIR)
sys.path.insert(0, os.path.join(_TESTS_DIR, "..", ".."))

import pytest
from sklearn.feature_extraction.text import CountVectorizer


@pytest.fixture
def dados_lda():
    docs = [
        "machine learning algorithm data",
        "deep neural network training",
        "natural language processing text",
        "computer vision image recognition",
        "reinforcement learning agent policy",
        "statistical analysis regression data",
        "python programming software development",
        "database sql query optimization",
        "cloud computing distributed systems",
        "cybersecurity encryption network security",
        "machine learning deep neural",
        "data science analysis statistics",
        "algorithm design complexity theory",
        "web development javascript framework",
        "operating system kernel process",
    ]
    vectorizer = CountVectorizer(max_features=50)
    matriz = vectorizer.fit_transform(docs)
    vocabulario = vectorizer.get_feature_names_out().tolist()
    return matriz, vocabulario, len(docs)


class TestLDA:

    def test_shape_distribuicao_documentos(self, dados_lda):
        from lda_modelo import ModeloLDA

        matriz, _, n_docs = dados_lda
        lda = ModeloLDA(num_topicos=5, params={"max_iter": 5, "random_state": 42})
        lda.treinar(matriz)

        distribuicao = lda.obter_distribuicao_documentos()
        assert distribuicao.shape == (n_docs, 5)

    def test_obter_topicos_retorna_lista_correta(self, dados_lda):
        from lda_modelo import ModeloLDA

        matriz, vocabulario, _ = dados_lda
        lda = ModeloLDA(num_topicos=3, params={"max_iter": 5, "random_state": 42})
        lda.treinar(matriz)

        topicos = lda.obter_topicos(vocabulario, top_n=5)
        assert len(topicos) == 3
        for topico in topicos:
            assert len(topico) == 5
            for palavra in topico:
                assert isinstance(palavra, str)

    def test_perplexidade_eh_float(self, dados_lda):
        from lda_modelo import ModeloLDA

        matriz, _, _ = dados_lda
        lda = ModeloLDA(num_topicos=3, params={"max_iter": 5, "random_state": 42})
        lda.treinar(matriz)

        perplexidade = lda.obter_perplexidade(matriz)
        assert isinstance(perplexidade, float)
        assert perplexidade >= 0

    def test_erro_se_nao_treinado(self, dados_lda):
        from lda_modelo import ModeloLDA

        _, vocabulario, _ = dados_lda
        lda = ModeloLDA(num_topicos=3)

        with pytest.raises(RuntimeError):
            lda.obter_topicos(vocabulario)

        with pytest.raises(RuntimeError):
            lda.obter_distribuicao_documentos()

    def test_obter_topicos_com_pesos_retorna_tuplas(self, dados_lda):
        from lda_modelo import ModeloLDA

        matriz, vocabulario, _ = dados_lda
        lda = ModeloLDA(num_topicos=3, params={"max_iter": 5, "random_state": 42})
        lda.treinar(matriz)

        topicos = lda.obter_topicos_com_pesos(vocabulario, top_n=5)
        assert len(topicos) == 3
        for topico in topicos:
            assert len(topico) == 5
            for palavra, peso in topico:
                assert isinstance(palavra, str)
                assert isinstance(peso, float)
                assert peso >= 0

    def test_obter_topicos_com_pesos_erro_se_nao_treinado(self, dados_lda):
        from lda_modelo import ModeloLDA

        _, vocabulario, _ = dados_lda
        lda = ModeloLDA(num_topicos=3)

        with pytest.raises(RuntimeError):
            lda.obter_topicos_com_pesos(vocabulario)
