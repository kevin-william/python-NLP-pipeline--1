import json
import os
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fase1_config import VOCAB_ANALYSIS_OUTPUT, OUTPUT_DIR
from logger import setup_logger

logger = setup_logger(__name__)


def analyze_vocabulary(tokens_raw, tokens_filtered):
    raw_words = [t["lemma"].lower() for t in tokens_raw if t["lemma"].strip()]
    filtered_words = [t["lemma"].lower() for t in tokens_filtered if t["lemma"].strip()]

    raw_unique = set(raw_words)
    filtered_unique = set(filtered_words)

    raw_counter = Counter(raw_words)
    filtered_counter = Counter(filtered_words)

    reduction_pct = round(
        (1 - len(filtered_unique) / len(raw_unique)) * 100, 2
    ) if raw_unique else 0

    analysis = {
        "vocab_raw_count": len(raw_unique),
        "vocab_filtered_count": len(filtered_unique),
        "vocab_reduction_percent": reduction_pct,
        "top_20_raw": raw_counter.most_common(20),
        "top_20_filtered": filtered_counter.most_common(20),
        "removed_words_sample": list(raw_unique - filtered_unique)[:50],
    }

    logger.info("Analise de vocabulario: raw=%d, filtered=%d, reducao=%.2f%%",
                analysis["vocab_raw_count"],
                analysis["vocab_filtered_count"],
                analysis["vocab_reduction_percent"])

    with open(VOCAB_ANALYSIS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    logger.info("Analise salva em: %s", VOCAB_ANALYSIS_OUTPUT)

    return analysis


def plot_pos_distribution(df, output_path=None):
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "pos_distribution.png")

    pos_counts = df["pos"].value_counts()

    plt.figure(figsize=(12, 6))
    pos_counts.plot(kind="bar", color="steelblue", edgecolor="black")
    plt.title("Distribuicao de POS Tags", fontsize=14)
    plt.xlabel("POS Tag", fontsize=12)
    plt.ylabel("Frequencia", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Grafico de distribuicao POS salvo em: %s", output_path)

    return pos_counts.to_dict()


def plot_frequency_comparison(raw_counter, filtered_counter, top_n=20, output_path=None):
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "freq_comparison.png")

    raw_top = raw_counter.most_common(top_n)
    filtered_top = filtered_counter.most_common(top_n)

    raw_words, raw_counts = zip(*raw_top) if raw_top else ([], [])
    filt_words, filt_counts = zip(*filtered_top) if filtered_top else ([], [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.barh(list(raw_words), list(raw_counts), color="steelblue")
    ax1.set_title(f"Top {top_n} Palavras (Antes)", fontsize=12)
    ax1.invert_yaxis()

    ax2.barh(list(filt_words), list(filt_counts), color="darkorange")
    ax2.set_title(f"Top {top_n} Palavras (Depois)", fontsize=12)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Grafico comparativo de frequencia salvo em: %s", output_path)
