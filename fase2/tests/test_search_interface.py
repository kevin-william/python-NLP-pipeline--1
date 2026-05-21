import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pytest
from similarity.cosine_search import CosineSearchEngine
from search_interface import start_search_interface


class FakePipeline:
    class FakeConfig:
        def get(self, key, default=None):
            if key == "TOP_K_RESULTS":
                return 5
            return default

    def __init__(self):
        self.search_engines = {}
        self.config = FakePipeline.FakeConfig()

    def search_text(self, method, query, top_k=10):
        return self.search_engines[method].search(np.array([1, 0, 0]), top_k)


def test_search_interface_imports():
    assert callable(start_search_interface)
