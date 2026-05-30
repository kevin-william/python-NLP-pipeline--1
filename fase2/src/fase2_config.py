import os
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords as _sw_mod
STOPWORDS_BOW_TFIDF = list(_sw_mod.words("portuguese"))

# Diretorio raiz da fase2 (usado para construir caminhos relativos do projeto).
DIRETORIO_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Caminho do parquet de entrada gerado pela fase1 para o recorte atual.
CAMINHO_PARQUET_ENTRADA = os.path.join(DIRETORIO_BASE, "input", "1100-artigos_wikipedia-formatados-v001_lemmatizacao.parquet")

# Pasta padrao para salvar logs e artefatos da fase2.
DIRETORIO_SAIDA = os.path.join(DIRETORIO_BASE, "output")

# Caminho do arquivo de log principal da fase2.
CAMINHO_LOG = os.path.join(DIRETORIO_SAIDA, "1100-artigos_wikipedia-formatados-v001_lemmatizacao.log")

# Ordem dos metodos de embedding a treinar e disponibilizar na busca.
METODOS_EMBEDDING = ["tfidf", "word2vec", "bow"]  # Pode ser uma lista com um ou mais metodos (ex.: ["bow"], ["tfidf", "word2vec"], etc.)

# Quantidade padrao de resultados retornados por consulta textual.
TOP_K_RESULTADOS = 5

# Parametros do CountVectorizer (Bag-of-Words).
PARAMS_BOW = {"max_features": 5000, "min_df": 1, "max_df": 0.8, "stop_words": STOPWORDS_BOW_TFIDF}

# Parametros do TfidfVectorizer.
PARAMS_TFIDF = {"max_features": 5000, "min_df": 1, "max_df": 0.8, "norm": "l2", "stop_words": STOPWORDS_BOW_TFIDF}

# Parametros de treino do Word2Vec (Gensim).
PARAMS_WORD2VEC = {"vector_size": 100, "window": 5, "min_count": 5, "epochs": 50, "seed": 42, "workers": 2}

# Habilita/desabilita a remocao de stopwords portuguesas durante o pre-processamento.
HABILITAR_REMOCAO_STOPWORDS = True

# POS tags permitidos na filtragem de tokens (whitelist). None ou lista vazia desabilita o filtro de POS.
POS_TAGS_PERMITIDOS = ["NOUN", "VERB", "ADJ", "ADV"]

# Habilita/desabilita a geracao do grafico t-SNE apos o treino.
HABILITAR_TSNE = True

# Parametros de reducao de dimensionalidade do t-SNE.
PARAMS_TSNE = {"n_components": 2, "perplexity": 5, "n_iter": 2000, "init": "pca", "random_state": 0}

# Parametros visuais do plot t-SNE salvo em arquivo.
PARAMS_PLOT_TSNE = {
    "figsize": (24, 18),
    "dpi": 600,
    "marker_size": 50,
    "annotate_fontsize": 7,
}

# Caminho final da imagem t-SNE gerada no pipeline.
CAMINHO_SAIDA_TSNE = os.path.join(DIRETORIO_SAIDA, "1100-artigos_wikipedia-formatados-v001_lemmatizacao.png")

# Habilita/desabilita o calculo de c-TF-IDF e geracao de WordCloud por categoria.
HABILITAR_CTFIDF = True

# Categoria padrao para artigos sem match nas regras de categorizacao.
CATEGORIA_PADRAO = "geral"

# Regras de categorizacao por palavra-chave no titulo (case-insensitive).
# Primeira regra que casar define a categoria do documento.
REGRAS_CATEGORIAS = [
    (
        ["programação", "linguagem", "python", "java", "c++", "ruby", "scala",
         "haskell", "lisp", "perl", "php", "cobol", "fortran", "software",
         "compilador", "framework", "paradigma", "objeto", "algoritmo",
         "recursão", "herança", "polimorfismo"],
        "programacao",
    ),
    (
        ["política", "politica", "palestina", "israel", "progressis",
         "liberalis", "esquerda", "direita", "governo", "democracia", "presidente"],
        "politica",
    ),
    (
        ["matemática", "matematica", "função", "cálculo", "lambda",
         "exponenciação", "adição", "subtração"],
        "matematica",
    ),
]

# Caminho para salvar o grid de WordClouds por categoria (c-TF-IDF).
CAMINHO_SAIDA_WORDCLOUD_CTFIDF = os.path.join(DIRETORIO_SAIDA, "1100-artigos_wikipedia-formatados-v001_ctfidf_wordclouds.png")
