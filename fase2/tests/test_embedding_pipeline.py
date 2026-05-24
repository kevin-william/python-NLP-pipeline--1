import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pandas as pd
import pytest
from embedding_pipeline import PipelineEmbeddings


@pytest.fixture
def sample_parquet(tmp_path):
    df = pd.DataFrame({
        "id_artigo": [1, 1, 1, 2, 2, 3, 3, 3],
        "id_token": [1, 2, 3, 1, 2, 1, 2, 3],
        "token": ["ola", "mundo", "!", "python", "linguagem", "processamento", "texto", "dados"],
        "pos": ["NOUN", "NOUN", "PUNCT", "NOUN", "NOUN", "NOUN", "NOUN", "NOUN"],
        "tag": ["N", "N", "PU", "N", "N", "N", "N", "N"],
        "lema": ["ola", "mundo", "!", "python", "linguagem", "processamento", "texto", "dados"],
        "relacao_dependencia": ["nsubj", "obj", "punct", "nsubj", "obj", "nsubj", "obj", "obj"],
        "token_cabeca": ["", "", "", "", "", "", "", ""],
        "entidade": ["", "", "", "", "", "", "", ""],
        "rotulo_entidade": ["", "", "", "", "", "", "", ""],
        "titulo": ["Doc1"] * 3 + ["Doc2"] * 2 + ["Doc3"] * 3,
        "url": ["http://1"] * 3 + ["http://2"] * 2 + ["http://3"] * 3,
    })
    parquet_path = tmp_path / "sample.parquet"
    df.to_parquet(parquet_path, index=False)
    return str(parquet_path)


class TestEmbeddingPipeline:
    def test_load_and_prepare(self, sample_parquet):
        config = {
            "METODOS_EMBEDDING": ["bow"],
            "PARAMS_BOW": {"max_features": 100, "min_df": 1},
            "TOP_K_RESULTADOS": 5,
            "HABILITAR_TSNE": False,
        }
        pipeline = PipelineEmbeddings(config, sample_parquet)
        pipeline._carregar_e_preparar()
        assert len(pipeline.documentos) == 3
        assert pipeline.titulos_documentos == ["Doc1", "Doc2", "Doc3"]

    def test_run_bow(self, sample_parquet):
        config = {
            "METODOS_EMBEDDING": ["bow"],
            "PARAMS_BOW": {"max_features": 100, "min_df": 1},
            "TOP_K_RESULTADOS": 5,
            "HABILITAR_TSNE": False,
        }
        pipeline = PipelineEmbeddings(config, sample_parquet)
        engines = pipeline.executar()
        assert "bow" in engines
        assert "bow" in pipeline.vetorizadores

    def test_run_multiple_methods(self, sample_parquet):
        config = {
            "METODOS_EMBEDDING": ["bow", "tfidf"],
            "PARAMS_BOW": {"max_features": 100, "min_df": 1},
            "PARAMS_TFIDF": {"max_features": 100, "min_df": 1},
            "TOP_K_RESULTADOS": 5,
            "HABILITAR_TSNE": False,
        }
        pipeline = PipelineEmbeddings(config, sample_parquet)
        engines = pipeline.executar()
        assert "bow" in engines
        assert "tfidf" in engines

    def test_search_text_bow(self, sample_parquet):
        config = {
            "METODOS_EMBEDDING": ["bow"],
            "PARAMS_BOW": {"max_features": 100, "min_df": 1},
            "TOP_K_RESULTADOS": 5,
            "HABILITAR_TSNE": False,
        }
        pipeline = PipelineEmbeddings(config, sample_parquet)
        pipeline.executar()

        results = pipeline.buscar_texto("bow", "mundo", top_k=2)
        assert len(results) > 0

    def test_search_text_unknown_method(self, sample_parquet):
        config = {
            "METODOS_EMBEDDING": ["bow"],
            "PARAMS_BOW": {"max_features": 100, "min_df": 1},
            "TOP_K_RESULTADOS": 5,
            "HABILITAR_TSNE": False,
        }
        pipeline = PipelineEmbeddings(config, sample_parquet)
        pipeline.executar()

        results = pipeline.buscar_texto("nonexistent", "query")
        assert results == []
