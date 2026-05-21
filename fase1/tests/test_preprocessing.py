import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest
from preprocessing import (
    get_nlp,
    get_stopwords,
    add_custom_stopwords,
    tokenize_article,
    remove_stopwords_from_tokens,
)


def test_get_nlp_returns_model():
    nlp = get_nlp()
    assert nlp is not None
    assert nlp.lang == "pt"


def test_get_stopwords_not_empty():
    stopwords = get_stopwords()
    assert len(stopwords) > 0
    assert "de" in stopwords or "o" in stopwords or "a" in stopwords


def test_add_custom_stopwords():
    add_custom_stopwords(["palavrateste"])
    stopwords = get_stopwords()
    assert "palavrateste" in stopwords


def test_tokenize_article():
    text = "O processamento de linguagem natural é fascinante."
    result = tokenize_article(text)
    assert "tokens" in result
    assert "sentences" in result
    assert len(result["tokens"]) > 0
    assert len(result["sentences"]) > 0
    token_keys = ["text", "lemma", "pos", "tag", "dep", "is_stop", "is_punct", "is_alpha"]
    for key in token_keys:
        assert key in result["tokens"][0]


def test_remove_stopwords_from_tokens():
    text = "O processamento de linguagem natural é fascinante."
    result = tokenize_article(text)
    filtered = remove_stopwords_from_tokens(result["tokens"])
    filtered_texts = [t["text"].lower() for t in filtered]
    assert "o" not in filtered_texts or "de" not in filtered_texts
    assert "processamento" in filtered_texts
