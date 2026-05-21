import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest
from vectorizers.bow_vectorizer import BowVectorizer


class TestBowVectorizer:
    def test_fit_transform(self):
        docs = ["ola mundo programacao", "python linguagem natural", "processamento texto python"]
        bow = BowVectorizer(max_features=100, min_df=1)
        matrix = bow.fit_transform(docs)
        assert matrix.shape[0] == 3
        assert matrix.shape[1] >= 6
        assert len(bow.get_feature_names()) >= 6

    def test_transform(self):
        docs = ["ola mundo", "python e legal"]
        bow = BowVectorizer()
        bow.fit_transform(docs)
        result = bow.transform(["mundo python"])
        assert result.shape[0] == 1

    def test_transform_before_fit(self):
        bow = BowVectorizer()
        with pytest.raises(RuntimeError):
            bow.transform(["teste"])

    def test_empty_documents(self):
        bow = BowVectorizer(min_df=1)
        with pytest.raises(ValueError):
            bow.fit_transform([""])

    def test_get_feature_names_before_fit(self):
        bow = BowVectorizer()
        assert bow.get_feature_names() == []
