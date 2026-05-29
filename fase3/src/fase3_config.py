import os

DIRETORIO_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAMINHO_ARTEFATO_FASE2 = os.path.join(DIRETORIO_BASE, "input", "fase2_artifact.lpf2")

DIRETORIO_SAIDA = os.path.join(DIRETORIO_BASE, "output")

CAMINHO_LOG = os.path.join(DIRETORIO_SAIDA, "fase3_pipeline.log")

DIRETORIO_PLOTS = os.path.join(DIRETORIO_SAIDA, "plots")

NUM_TOPICOS = 10

TOP_N_PALAVRAS = 10

PARAMS_LSA = {"random_state": 42}

PARAMS_LDA = {
    "max_iter": 20,
    "learning_method": "online",
    "random_state": 42,
    "n_jobs": -1,
}

PARAMS_NMF = {
    "init": "nndsvda",
    "random_state": 42,
    "max_iter": 500,
}

PARAMS_BOW_RECONSTRUCAO = {"max_features": 5000, "min_df": 1}
