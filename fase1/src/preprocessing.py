import spacy
from fase1_config import SPACY_MODEL
from logger import setup_logger

logger = setup_logger(__name__)

_custom_stopwords = set()
_nlp_instance = None


def get_nlp():
    global _nlp_instance
    if _nlp_instance is None:
        logger.info("Carregando modelo spaCy: %s", SPACY_MODEL)
        _nlp_instance = spacy.load(SPACY_MODEL)
        logger.info("Modelo carregado com sucesso")
    return _nlp_instance


def get_stopwords():
    nlp = get_nlp()
    return nlp.Defaults.stop_words | _custom_stopwords


def add_custom_stopwords(words):
    for w in words:
        _custom_stopwords.add(w.lower())
    logger.info("Stopwords customizadas adicionadas: %s", words)


def remove_custom_stopwords(words):
    for w in words:
        _custom_stopwords.discard(w.lower())
    logger.info("Stopwords customizadas removidas: %s", words)


def tokenize_article(article_text):
    nlp = get_nlp()
    doc = nlp(article_text)
    tokens = [
        {
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "dep": token.dep_,
            "is_stop": token.is_stop,
            "is_punct": token.is_punct,
            "is_alpha": token.is_alpha,
        }
        for token in doc
    ]
    sentences = [sent.text for sent in doc.sents]
    return {"tokens": tokens, "sentences": sentences}


def remove_stopwords_from_tokens(tokens):
    stopwords = get_stopwords()
    filtered = [t for t in tokens if t["text"].lower() not in stopwords and not t["is_punct"]]
    return filtered
