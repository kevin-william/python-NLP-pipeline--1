import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pandas as pd
import pytest
from embedding_pipeline import EmbeddingPipeline


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "artigo_id": [1, 1, 1, 2, 2, 3, 3, 3],
        "token_id": [1, 2, 3, 1, 2, 1, 2, 3],
        "token": ["ola", "mundo", "!", "python", "linguagem", "processamento", "texto", "dados"],
        "pos": ["NOUN", "NOUN", "PUNCT", "NOUN", "NOUN", "NOUN", "NOUN", "NOUN"],
        "tag": ["N", "N", "PU", "N", "N", "N", "N", "N"],
        "lemma": ["ola", "mundo", "!", "python", "linguagem", "processamento", "texto", "dados"],
        "dep_rel": ["nsubj", "obj", "punct", "nsubj", "obj", "nsubj", "obj", "obj"],
        "head_token": ["", "", "", "", "", "", "", ""],
        "entity": ["", "", "", "", "", "", "", ""],
        "entity_label": ["", "", "", "", "", "", "", ""],
        "title": ["Doc1"] * 3 + ["Doc2"] * 2 + ["Doc3"] * 3,
        "url": ["http://1"] * 3 + ["http://2"] * 2 + ["http://3"] * 3,
    })
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return str(csv_path)


class TestEmbeddingPipeline:
    def test_load_and_prepare(self, sample_csv):
        config = {
            "EMBEDDING_METHODS": ["bow"],
            "BOW_PARAMS": {"max_features": 100, "min_df": 1},
            "TOP_K_RESULTS": 5,
            "ENABLE_TSNE": False,
        }
        pipeline = EmbeddingPipeline(config, sample_csv)
        pipeline._load_and_prepare()
        assert len(pipeline.documents) == 3
        assert pipeline.document_titles == ["Doc1", "Doc2", "Doc3"]

    def test_run_bow(self, sample_csv):
        config = {
            "EMBEDDING_METHODS": ["bow"],
            "BOW_PARAMS": {"max_features": 100, "min_df": 1},
            "TOP_K_RESULTS": 5,
            "ENABLE_TSNE": False,
        }
        pipeline = EmbeddingPipeline(config, sample_csv)
        engines = pipeline.run()
        assert "bow" in engines
        assert "bow" in pipeline.vectorizers

    def test_run_multiple_methods(self, sample_csv):
        config = {
            "EMBEDDING_METHODS": ["bow", "tfidf"],
            "BOW_PARAMS": {"max_features": 100, "min_df": 1},
            "TFIDF_PARAMS": {"max_features": 100, "min_df": 1},
            "TOP_K_RESULTS": 5,
            "ENABLE_TSNE": False,
        }
        pipeline = EmbeddingPipeline(config, sample_csv)
        engines = pipeline.run()
        assert "bow" in engines
        assert "tfidf" in engines

    def test_search_text_bow(self, sample_csv):
        config = {
            "EMBEDDING_METHODS": ["bow"],
            "BOW_PARAMS": {"max_features": 100, "min_df": 1},
            "TOP_K_RESULTS": 5,
            "ENABLE_TSNE": False,
        }
        pipeline = EmbeddingPipeline(config, sample_csv)
        pipeline.run()

        results = pipeline.search_text("bow", "mundo", top_k=2)
        assert len(results) > 0

    def test_search_text_unknown_method(self, sample_csv):
        config = {
            "EMBEDDING_METHODS": ["bow"],
            "BOW_PARAMS": {"max_features": 100, "min_df": 1},
            "TOP_K_RESULTS": 5,
            "ENABLE_TSNE": False,
        }
        pipeline = EmbeddingPipeline(config, sample_csv)
        pipeline.run()

        results = pipeline.search_text("nonexistent", "query")
        assert results == []
