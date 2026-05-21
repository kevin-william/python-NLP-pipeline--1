import pandas as pd
from preprocessing import get_nlp
from fase1_config import BATCH_SIZE
from logger import setup_logger

logger = setup_logger(__name__)


def process_articles_batch(articles):
    nlp = get_nlp()
    texts = [a["content"] for a in articles]
    logger.info("Processando %d artigos em lote (batch_size=%d)...", len(texts), BATCH_SIZE)

    rows = []

    for artigo_id, (article, doc) in enumerate(
        zip(articles, nlp.pipe(texts, batch_size=BATCH_SIZE)), start=1
    ):
        for token_id, token in enumerate(doc, start=1):
            entity = ""
            entity_label = ""
            if token.ent_iob_ == "B" or token.ent_iob_ == "I":
                entity = token.ent_type_ or ""
                entity_label = token.ent_type_ or ""
            rows.append(
                {
                    "artigo_id": artigo_id,
                    "token_id": token_id,
                    "token": token.text,
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "lemma": token.lemma_,
                    "dep_rel": token.dep_,
                    "head_token": token.head.text if token.head else "",
                    "entity": entity,
                    "entity_label": entity_label,
                    "title": article["title"],
                    "url": article["url"],
                }
            )
        logger.info(
            "Artigo %d processado: '%s' -> %d tokens",
            artigo_id,
            article["title"],
            token_id,
        )

    df = pd.DataFrame(rows)
    logger.info("DataFrame criado: %d linhas, %d colunas", len(df), len(df.columns))
    return df
