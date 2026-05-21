import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from collections import Counter
from fase1_config import CSV_OUTPUT, WORDCLOUD_OUTPUT, OUTPUT_DIR
from logger import setup_logger
from corpus_loader import load_articles, get_corpus_statistics
from preprocessing import get_stopwords, add_custom_stopwords
from pos_tagger import process_articles_batch
from wordcloud_gen import generate_wordcloud
from vocab_analysis import (
    analyze_vocabulary,
    plot_pos_distribution,
    plot_frequency_comparison,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

logger = setup_logger("nlp_pipeline")


def main():
    logger.info("=" * 60)
    logger.info("INICIANDO PIPELINE DE NLP - Wikipedia Articles")
    logger.info("=" * 60)

    # Step 1: Load corpus
    logger.info("[ETAPA 1] Carregamento e inspecao do corpus")
    articles = load_articles()
    stats = get_corpus_statistics(articles)
    logger.info("Estatisticas: %s", stats)

    # Step 2: POS tagging with spaCy
    logger.info("[ETAPA 2] POS Tagging com spaCy")
    df = process_articles_batch(articles)
    df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8")
    df.to_parquet(CSV_OUTPUT.replace(".csv", ".parquet"), index=False)
    logger.info("DataFrame salvo: %s (CSV e Parquet)", CSV_OUTPUT)
    logger.info("CSV exportado: %s", CSV_OUTPUT)

    # Step 3: Stopword removal + vocabulary analysis
    logger.info("[ETAPA 3] Analise de vocabulario (stopwords)")
    stopwords = get_stopwords()
    logger.info("Stopwords carregadas: %d palavras", len(stopwords))

    all_tokens = df.to_dict("records")

    def is_not_stop(token_row):
        text = str(token_row.get("lemma", "")).lower()
        pos = str(token_row.get("pos", ""))
        return text not in stopwords and pos != "PUNCT" and text.strip()

    tokens_raw_info = [
        {"text": str(t.get("token", "")), "lemma": str(t.get("lemma", "")), "pos": str(t.get("pos", ""))}
        for t in all_tokens
    ]
    tokens_filtered = [
        {"text": str(t.get("token", "")), "lemma": str(t.get("lemma", "")), "pos": str(t.get("pos", ""))}
        for t in all_tokens if is_not_stop(t)
    ]

    raw_counter = Counter(t["lemma"].lower() for t in tokens_raw_info if t["lemma"].strip())
    filtered_counter = Counter(t["lemma"].lower() for t in tokens_filtered if t["lemma"].strip())

    analyze_vocabulary(tokens_raw_info, tokens_filtered)

    # Step 4: POS distribution
    logger.info("[ETAPA 4] Distribuicao de POS Tags")
    plot_pos_distribution(df)

    # Step 5: Frequency comparison
    logger.info("[ETAPA 5] Grafico comparativo de frequencia")
    plot_frequency_comparison(raw_counter, filtered_counter)

    # Step 6: WordCloud
    logger.info("[ETAPA 6] Geracao de WordCloud")
    generate_wordcloud(tokens_filtered)
    generate_wordcloud(tokens_filtered, include_stopwords=False)

    logger.info("=" * 60)
    logger.info("PIPELINE CONCLUIDO COM SUCESSO")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
