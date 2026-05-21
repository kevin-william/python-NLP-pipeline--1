import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest
from corpus_loader import load_articles, get_corpus_statistics


def test_load_articles_count():
    articles = load_articles()
    assert len(articles) > 0, "Deveria carregar ao menos 1 artigo"


def test_article_structure():
    articles = load_articles()
    for article in articles:
        assert "title" in article
        assert "url" in article
        assert "content" in article
        assert isinstance(article["title"], str) and len(article["title"]) > 0
        assert isinstance(article["url"], str) and len(article["url"]) > 0
        assert isinstance(article["content"], str)


def test_article_titles():
    articles = load_articles()
    titles = [a["title"] for a in articles]
    assert "Processamento de linguagem natural" in titles


def test_get_corpus_statistics():
    articles = load_articles()
    stats = get_corpus_statistics(articles)
    assert stats["total_articles"] == len(articles)
    assert stats["total_characters"] > 0
    assert stats["avg_characters_per_article"] > 0
