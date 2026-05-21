import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest
from corpus_loader import load_articles
from preprocessing import (
    tokenize_article,
    remove_stopwords_from_tokens,
    get_stopwords,
)
from vocab_analysis import analyze_vocabulary
from collections import Counter


def test_full_mini_pipeline():
    articles = load_articles()
    first_article = articles[0]
    result = tokenize_article(first_article["content"])
    tokens = result["tokens"]
    filtered = remove_stopwords_from_tokens(tokens)

    raw_lemma_counter = Counter(t["lemma"].lower() for t in tokens if t["lemma"].strip())
    filtered_lemma_counter = Counter(t["lemma"].lower() for t in filtered if t["lemma"].strip())

    assert len(raw_lemma_counter) > 0
    assert len(filtered_lemma_counter) > 0
    assert len(filtered_lemma_counter) <= len(raw_lemma_counter)


def test_stopword_removal_effect():
    articles = load_articles()
    content = articles[0]["content"]
    result = tokenize_article(content)
    tokens = result["tokens"]
    filtered = remove_stopwords_from_tokens(tokens)
    stopwords = get_stopwords()

    stop_tokens_before = sum(1 for t in tokens if t["text"].lower() in stopwords)
    stop_tokens_after = sum(1 for t in filtered if t["text"].lower() in stopwords)

    assert stop_tokens_after == 0, "Nao deveria haver stopwords apos remocao"
    assert len(filtered) <= len(tokens), "Filtro deveria reduzir ou manter tokens"


def test_vocab_analysis_output():
    articles = load_articles()
    content = articles[0]["content"]
    result = tokenize_article(content)
    tokens = result["tokens"]

    tokens_info = [
        {"text": t["text"], "lemma": t["lemma"], "pos": t["pos"]}
        for t in tokens
    ]
    filtered = remove_stopwords_from_tokens(tokens)
    filtered_info = [
        {"text": t["text"], "lemma": t["lemma"], "pos": t["pos"]}
        for t in filtered
    ]

    analysis = analyze_vocabulary(tokens_info, filtered_info)
    assert "vocab_raw_count" in analysis
    assert "vocab_filtered_count" in analysis
    assert "vocab_reduction_percent" in analysis
    assert analysis["vocab_filtered_count"] <= analysis["vocab_raw_count"]
