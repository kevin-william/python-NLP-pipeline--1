import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from preprocessing import get_stopwords
from fase1_config import WORDCLOUD_OUTPUT
from logger import setup_logger

logger = setup_logger(__name__)


def generate_wordcloud(
    tokens,
    width=1200,
    height=600,
    max_words=200,
    colormap="viridis",
    background_color="white",
    include_stopwords=False,
    output_path=None,
):
    if output_path is None:
        output_path = WORDCLOUD_OUTPUT

    if include_stopwords:
        words_text = " ".join(t["text"] for t in tokens)
    else:
        stopwords = get_stopwords()
        words_text = " ".join(
            t["lemma"] for t in tokens
            if t["lemma"].lower() not in stopwords and t["lemma"].strip()
        )

    if not words_text.strip():
        logger.warning("Nenhum token para gerar wordcloud")
        return

    wc = WordCloud(
        prefer_horizontal=1,
        width=width,
        height=height,
        max_words=max_words,
        colormap=colormap,
        background_color=background_color,
    ).generate(words_text)

    plt.figure(figsize=(16, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("WordCloud salva em: %s", output_path)


def generate_wordcloud_by_pos(tokens, pos_filter, output_path=None):
    if output_path is None:
        output_path = WORDCLOUD_OUTPUT

    filtered = [t for t in tokens if t.get("pos") == pos_filter]
    logger.info("Gerando wordcloud para POS=%s (%d tokens)", pos_filter, len(filtered))
    generate_wordcloud(filtered, output_path=output_path)
