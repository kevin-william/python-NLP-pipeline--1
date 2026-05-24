import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pytest
from similarity.cosine_search import MotorBuscaCosseno
from search_interface import iniciar_interface_busca


class FakePipeline:
    class FakeConfig:
        def get(self, key, default=None):
            if key == "TOP_K_RESULTADOS":
                return 5
            return default

    def __init__(self):
        self.motores_busca = {}
        self.configuracoes = FakePipeline.FakeConfig()

    def buscar_texto(self, method, query, top_k=10):
        return self.motores_busca[method].search(np.array([1, 0, 0]), top_k)


def test_search_interface_imports():
    assert callable(iniciar_interface_busca)
