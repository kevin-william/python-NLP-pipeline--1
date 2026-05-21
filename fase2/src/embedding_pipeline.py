import os
import numpy as np
import pandas as pd
from logger import setup_logger
from vectorizers.bow_vectorizer import BowVectorizer
from vectorizers.tfidf_vectorizer import TfidfVectorizerWrapper
from vectorizers.word2vec_vectorizer import Word2VecWrapper
from similarity.cosine_search import CosineSearchEngine
from visualization.tsne_plot import TSNEVisualizer

logger = setup_logger(__name__)


class EmbeddingPipeline:
    def __init__(self, config, input_csv):
        self.config = config
        self.input_csv = input_csv
        self.search_engines = {}
        self.vectorizers = {}
        self.documents = []
        self.document_titles = []
        self.tokenized_docs = []

    def _load_and_prepare(self):
        logger.info("Carregando dados de: %s", self.input_csv)
        df = pd.read_csv(self.input_csv, encoding="utf-8")
        logger.info("CSV carregado: %d linhas, %d colunas", len(df), len(df.columns))

        grouped = df.groupby("artigo_id")

        for artigo_id, group in grouped:
            title = str(group["title"].iloc[0])
            lemmas = group["lemma"].dropna()

            filtered_tokens = []
            for idx, lemma_series in lemmas.items():
                lemma = str(lemma_series).strip().lower()
                pos = str(group.loc[idx, "pos"]) if idx in group.index else ""
                if lemma and pos != "PUNCT":
                    filtered_tokens.append(lemma)

            doc_text = " ".join(filtered_tokens)
            self.documents.append(doc_text)
            self.document_titles.append(title)
            self.tokenized_docs.append(filtered_tokens)

        logger.info("Documentos preparados: %d artigos", len(self.documents))
        return self

    def run(self):
        self._load_and_prepare()

        methods = self.config["EMBEDDING_METHODS"]
        logger.info("Metodos configurados: %s", methods)

        for method in methods:
            logger.info("--- Treinando metodo: %s ---", method)
            engine = None
            vectorizer = None

            if method == "bow":
                engine, vectorizer = self._train_bow()
            elif method == "tfidf":
                engine, vectorizer = self._train_tfidf()
            elif method == "word2vec":
                engine, vectorizer = self._train_word2vec()
            else:
                logger.warning("Metodo desconhecido: %s", method)
                continue

            if engine and vectorizer:
                self.search_engines[method] = engine
                self.vectorizers[method] = vectorizer

        if self.config.get("ENABLE_TSNE", False):
            self._run_tsne()

        logger.info("Treinamento concluido. %d search engines disponiveis.", len(self.search_engines))
        return self.search_engines

    def _train_bow(self):
        params = self.config.get("BOW_PARAMS", {})
        vectorizer = BowVectorizer(**params)
        doc_vectors = vectorizer.fit_transform(self.documents)
        engine = CosineSearchEngine("bow", self.documents)
        engine.fit(doc_vectors)
        return engine, vectorizer

    def _train_tfidf(self):
        params = self.config.get("TFIDF_PARAMS", {})
        vectorizer = TfidfVectorizerWrapper(**params)
        doc_vectors = vectorizer.fit_transform(self.documents)
        engine = CosineSearchEngine("tfidf", self.documents)
        engine.fit(doc_vectors)
        return engine, vectorizer

    def _train_word2vec(self):
        params = self.config.get("WORD2VEC_PARAMS", {})
        model = Word2VecWrapper(**params)
        model.fit(self.tokenized_docs)
        doc_vectors = model.get_mean_document_embeddings(self.tokenized_docs)
        engine = CosineSearchEngine("word2vec", self.documents)
        engine.fit(doc_vectors)
        return engine, model

    def _run_tsne(self):
        if not self.search_engines:
            logger.warning("Nenhum search engine para t-SNE")
            return

        first_method = list(self.search_engines.keys())[0]
        engine = self.search_engines[first_method]
        vectors = engine.doc_vectors

        if hasattr(vectors, "toarray"):
            vectors = vectors.toarray()
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)

        tsne_params = self.config.get("TSNE_PARAMS", {})
        tsne_plot_params = self.config.get("TSNE_PLOT_PARAMS", {})
        tsne_output = self.config.get("TSNE_OUTPUT", os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "tsne_plot.png"
        ))

        visualizer = TSNEVisualizer(**tsne_params)
        embeddings_2d = visualizer.fit_transform(vectors)
        visualizer.plot(embeddings_2d, self.document_titles, tsne_output, first_method, **tsne_plot_params)

    def search_text(self, method, query_text, top_k=10):
        if method not in self.search_engines:
            logger.warning("Metodo '%s' nao disponivel", method)
            return []

        if method not in self.vectorizers:
            logger.warning("Vectorizer para '%s' nao disponivel", method)
            return []

        vectorizer = self.vectorizers[method]

        if method == "word2vec":
            words = query_text.strip().lower().split()
            query_vector = vectorizer.get_sentence_vector(words)
        else:
            query_vector = vectorizer.transform([query_text])
            if hasattr(query_vector, "toarray"):
                query_vector = query_vector.toarray().flatten()

        return self.search_engines[method].search(query_vector, top_k)
