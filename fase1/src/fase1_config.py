import os
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(BASE_DIR, "input", "artigos_wikipedia.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_FILE = os.path.join(OUTPUT_DIR, "nlp_pipeline.log")
CSV_OUTPUT = os.path.join(OUTPUT_DIR, "artigos_anotacao_lg.csv")
WORDCLOUD_OUTPUT = os.path.join(OUTPUT_DIR, "wordcloud.png")
VOCAB_ANALYSIS_OUTPUT = os.path.join(OUTPUT_DIR, "vocabulario_analise.json")

SPACY_MODEL = "pt_core_news_lg"
BATCH_SIZE = 10

ARTICLE_START_MARKER = "===== ARTICLE START ====="
ARTICLE_END_MARKER = "===== ARTICLE END ====="
