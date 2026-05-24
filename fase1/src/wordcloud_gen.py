import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from preprocessing import obter_stopwords
from fase1_config import CAMINHO_NUVEM_PALAVRAS
from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)


def gerar_nuvem_palavras(
    tokens,
    largura=1200,
    altura=600,
    maximo_palavras=200,
    paleta_cores="viridis",
    cor_fundo="white",
    incluir_stopwords=False,
    caminho_saida=None,
):
    if caminho_saida is None:
        caminho_saida = CAMINHO_NUVEM_PALAVRAS

    def _obter_texto_token(token):
        return str(token.get("processado", token.get("lema", token.get("texto", token.get("text", "")))))

    if incluir_stopwords:
        texto_palavras = " ".join(
            str(token.get("texto", token.get("text", ""))) for token in tokens
        )
    else:
        stopwords_atuais = obter_stopwords()
        texto_palavras = " ".join(
            texto_token for token in tokens
            if (texto_token := _obter_texto_token(token)).lower() not in stopwords_atuais and texto_token.strip()
        )

    if not texto_palavras.strip():
        logger.warning("Nenhum token para gerar wordcloud")
        return

    nuvem = WordCloud(
        prefer_horizontal=1,
        width=largura,
        height=altura,
        max_words=maximo_palavras,
        colormap=paleta_cores,
        background_color=cor_fundo,
    ).generate(texto_palavras)

    plt.figure(figsize=(16, 8))
    plt.imshow(nuvem, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(caminho_saida, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("WordCloud salva em: %s", caminho_saida)


def gerar_nuvem_palavras_por_pos(tokens, filtro_pos, caminho_saida=None):
    if caminho_saida is None:
        caminho_saida = CAMINHO_NUVEM_PALAVRAS

    tokens_filtrados = [token for token in tokens if token.get("pos") == filtro_pos]
    logger.info("Gerando wordcloud para POS=%s (%d tokens)", filtro_pos, len(tokens_filtrados))
    gerar_nuvem_palavras(tokens_filtrados, caminho_saida=caminho_saida)
