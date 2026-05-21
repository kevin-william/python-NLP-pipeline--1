import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest
from pos_tagger import process_articles_batch


def test_process_articles_batch():
    articles = [
        {
            "title": "Teste",
            "url": "https://example.com",
            "content": "O processamento de linguagem natural é uma área fascinante da inteligência artificial.",
        }
    ]
    df = process_articles_batch(articles)
    assert len(df) > 0
    expected_columns = [
        "artigo_id", "token_id", "token", "pos", "tag",
        "lemma", "dep_rel", "head_token", "entity", "entity_label",
        "title", "url",
    ]
    for col in expected_columns:
        assert col in df.columns, f"Coluna '{col}' ausente"
    assert df["artigo_id"].iloc[0] == 1
    assert df["title"].iloc[0] == "Teste"
    assert df["url"].iloc[0] == "https://example.com"


def test_pos_tags_not_empty():
    articles = [
        {
            "title": "Teste 2",
            "url": "https://example.com/2",
            "content": "O gato preto correu rapidamente.",
        }
    ]
    df = process_articles_batch(articles)
    pos_values = df["pos"].unique()
    assert len(pos_values) > 0, "Deveria ter POS tags"
    assert "VERB" in pos_values or "NOUN" in pos_values


def test_lemma_present():
    articles = [
        {
            "title": "Teste 3",
            "url": "https://example.com/3",
            "content": "Os computadores processam dados.",
        }
    ]
    df = process_articles_batch(articles)
    lemmas = df["lemma"].dropna().tolist()
    assert len(lemmas) > 0
    lemmas_lower = [l.lower() for l in lemmas]
    assert any("computador" in l for l in lemmas_lower)
