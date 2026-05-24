import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest
from vectorizers.tfidf_vectorizer import VetorizadorTfidf


class TestVetorizadorTfidf:
    def test_fit_transform(self):
        docs = ["ola mundo programacao", "python linguagem natural", "processamento texto python"]
        tfidf = VetorizadorTfidf(max_features=100, min_df=1)
        matrix = tfidf.fit_transform(docs)
        assert matrix.shape[0] == 3
        assert matrix.shape[1] >= 6

    def test_transform(self):
        docs = ["ola mundo", "python e legal"]
        tfidf = VetorizadorTfidf()
        tfidf.fit_transform(docs)
        result = tfidf.transform(["mundo python"])
        assert result.shape[0] == 1

    def test_transform_before_fit(self):
        tfidf = VetorizadorTfidf()
        with pytest.raises(RuntimeError):
            tfidf.transform(["teste"])

    def test_tfidf_values_positive(self):
        docs = ["ola mundo", "python e legal"]
        tfidf = VetorizadorTfidf()
        matrix = tfidf.fit_transform(docs)
        assert matrix.data.min() >= 0.0
