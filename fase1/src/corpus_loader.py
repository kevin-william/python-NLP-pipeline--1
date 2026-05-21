import re
from fase1_config import INPUT_FILE, ARTICLE_START_MARKER, ARTICLE_END_MARKER
from logger import setup_logger

logger = setup_logger(__name__)


def load_articles(filepath=None):
    if filepath is None:
        filepath = INPUT_FILE

    logger.info("Carregando artigos de: %s", filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    articles = []
    pattern = re.compile(
        re.escape(ARTICLE_START_MARKER)
        + r"\s*"
        + r"Title:\s*(.+?)$\s*"
        + r"URL:\s*(.+?)$\s*"
        + re.escape("=========================")
        + r"\s*"
        + r"(.*?)"
        + r"\s*"
        + re.escape(ARTICLE_END_MARKER),
        re.MULTILINE | re.DOTALL,
    )

    for match in pattern.finditer(text):
        title = match.group(1).strip()
        url = match.group(2).strip()
        content = match.group(3).strip()
        articles.append({"title": title, "url": url, "content": content})
        logger.info("Artigo carregado: '%s' (%d chars)", title, len(content))

    logger.info("Total de artigos carregados: %d", len(articles))
    return articles


def get_corpus_statistics(articles):
    total = len(articles)
    lengths = [len(a["content"]) for a in articles]
    total_chars = sum(lengths)
    avg_chars = total_chars / total if total > 0 else 0
    min_chars = min(lengths) if lengths else 0
    max_chars = max(lengths) if lengths else 0

    stats = {
        "total_articles": total,
        "total_characters": total_chars,
        "avg_characters_per_article": round(avg_chars, 2),
        "min_characters": min_chars,
        "max_characters": max_chars,
    }

    logger.info("Estatisticas do corpus: %s", stats)
    return stats
