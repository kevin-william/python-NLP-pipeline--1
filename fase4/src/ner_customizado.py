"""
Anotacao de exemplos e treinamento de um modelo NER customizado com spaCy.

Demonstra o ciclo completo: anotar -> treinar -> avaliar -> comparar.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd

from fase4_config import (
    CAMINHO_MODELO_CUSTOMIZADO,
    EPOCAS_TREINO_NER,
    NUMERO_EXEMPLOS_TREINO,
    TAXA_APRENDIZADO_NER,
)
from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)


def converter_entidades_para_formato_treino(
    parquet_df: pd.DataFrame,
    documentos: List[str],
) -> List[Tuple[str, Dict]]:
    """
    Converte entidades do parquet da Fase 1 para o formato de treino do spaCy:
    (texto, {"entities": [(inicio, fim, label)]})
    """
    dados_treino: List[Tuple[str, Dict]] = []

    # Agrupa entidades por id_artigo se disponivel
    if "id_artigo" not in parquet_df.columns:
        logger.warning("Coluna 'id_artigo' ausente no parquet — usando primeira amostra de documentos")
        return _anotar_com_regex(documentos[:NUMERO_EXEMPLOS_TREINO])

    for idx, texto in enumerate(documentos[:NUMERO_EXEMPLOS_TREINO]):
        sub = parquet_df[parquet_df["id_artigo"] == idx]
        entidades_anotadas: List[Tuple[int, int, str]] = []
        texto_lower = texto.lower()
        for _, row in sub.iterrows():
            rotulo = str(row.get("rotulo_entidade", "")).strip()
            token_text = str(row.get("entidade", row.get("token", ""))).strip()
            if not rotulo or not token_text or rotulo in ("", "nan", "None"):
                continue
            pos = texto_lower.find(token_text.lower())
            if pos >= 0:
                entidades_anotadas.append((pos, pos + len(token_text), rotulo))
        if entidades_anotadas:
            dados_treino.append((texto, {"entities": entidades_anotadas}))

    if not dados_treino:
        logger.info("Nenhum dado de treino obtido do parquet — usando anotacao automatica")
        dados_treino = _anotar_com_regex(documentos[:NUMERO_EXEMPLOS_TREINO])

    return dados_treino


def _anotar_com_regex(documentos: List[str]) -> List[Tuple[str, Dict]]:
    """
    Anotacao automatica usando spaCy como anotador (pseudo-ground-truth).
    """
    try:
        from ner_analysis import carregar_modelo_spacy
        nlp = carregar_modelo_spacy()
    except Exception:
        return []

    dados: List[Tuple[str, Dict]] = []
    for texto in documentos:
        doc = nlp(texto[:5000])
        entidades = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        if entidades:
            dados.append((texto[:5000], {"entities": entidades}))
    return dados


def anotar_exemplos_manual(
    documentos: List[str], n_exemplos: int = NUMERO_EXEMPLOS_TREINO
) -> List[Tuple[str, Dict]]:
    """
    Usa o modelo pre-treinado do spaCy para gerar anotacoes de treino.
    """
    return _anotar_com_regex(documentos[:n_exemplos])


def treinar_ner_customizado(
    dados_treino: List[Tuple[str, Dict]],
    modelo_base: str = "pt_core_news_lg",
    epochs: int = EPOCAS_TREINO_NER,
    learning_rate: float = TAXA_APRENDIZADO_NER,
    caminho_saida: Optional[str] = None,
):
    """
    Treina (fine-tune) um modelo NER simples com spaCy.
    Retorna o modelo treinado (Language).
    """
    import spacy
    from spacy.training import Example
    from spacy.util import compounding, minibatch

    if caminho_saida is None:
        caminho_saida = os.path.join(CAMINHO_MODELO_CUSTOMIZADO, "modelo_ner")

    if not dados_treino:
        logger.warning("Nenhum dado de treino disponivel — NER customizado nao sera treinado")
        try:
            return spacy.load(modelo_base)
        except OSError:
            import spacy
            return spacy.blank("pt")

    modelo_carregado = True
    try:
        nlp = spacy.load(modelo_base)
    except OSError:
        # "xx" e o modelo multilingue em branco que nao requer lookups especificos
        nlp = spacy.blank("xx")
        modelo_carregado = False
        logger.warning("Modelo base '%s' nao encontrado — treinando do zero (lang=xx)", modelo_base)

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Adiciona labels
    for _, annotations in dados_treino:
        for _, _, label in annotations.get("entities", []):
            ner.add_label(label)

    # Treina apenas o componente NER
    exemplos = []
    for texto, anotacoes in dados_treino:
        try:
            doc = nlp.make_doc(texto)
            ex = Example.from_dict(doc, anotacoes)
            exemplos.append(ex)
        except Exception as e:
            logger.debug("Exemplo ignorado: %s", e)

    if not exemplos:
        logger.warning("Nenhum exemplo valido para treino — retornando modelo base")
        return nlp

    with nlp.select_pipes(enable=["ner"]):
        # resume_training() nao chama initialize() nos componentes,
        # evitando o erro de lookups ausentes para 'pt'.
        # Para modelo em branco usa initialize() com get_examples vazio.
        if modelo_carregado:
            optimizer = nlp.resume_training()
        else:
            optimizer = nlp.initialize(get_examples=lambda: exemplos)
        optimizer.learn_rate = learning_rate
        for epoch in range(epochs):
            random.shuffle(exemplos)
            losses: Dict = {}
            batches = list(minibatch(exemplos, size=compounding(4.0, 32.0, 1.001)))
            for batch in batches:
                nlp.update(batch, drop=0.3, losses=losses)
            if (epoch + 1) % 5 == 0:
                logger.info("Epoch %d/%d — loss NER: %.4f", epoch + 1, epochs, losses.get("ner", 0))

    os.makedirs(caminho_saida, exist_ok=True)
    nlp.to_disk(caminho_saida)
    logger.info("Modelo NER customizado salvo em: %s", caminho_saida)
    return nlp


def avaliar_modelo_ner(
    modelo, dados_teste: List[Tuple[str, Dict]]
) -> Dict[str, float]:
    """
    Calcula precision, recall e f1-score do modelo NER em dados_teste.
    """
    if not dados_teste:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    from spacy.training import Example

    tp = 0
    fp = 0
    fn = 0
    for texto, anotacoes in dados_teste:
        try:
            doc_pred = modelo(texto)
            pred_ents = set(
                (ent.start_char, ent.end_char, ent.label_) for ent in doc_pred.ents
            )
            gold_ents = set(
                (s, e, l) for s, e, l in anotacoes.get("entities", [])
            )
            tp += len(pred_ents & gold_ents)
            fp += len(pred_ents - gold_ents)
            fn += len(gold_ents - pred_ents)
        except Exception as e:
            logger.debug("Erro na avaliacao: %s", e)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def comparar_modelos(
    modelo_pretreinado,
    modelo_customizado,
    textos: List[str],
) -> pd.DataFrame:
    """
    Compara entidades extraidas pelos dois modelos nos mesmos textos.
    """
    registros: list = []
    for idx, texto in enumerate(textos[:20]):
        doc_pre = modelo_pretreinado(texto[:5000])
        doc_cust = modelo_customizado(texto[:5000])
        pre_ents = {(ent.text, ent.label_) for ent in doc_pre.ents}
        cust_ents = {(ent.text, ent.label_) for ent in doc_cust.ents}
        ambos = pre_ents & cust_ents
        so_pre = pre_ents - cust_ents
        so_cust = cust_ents - pre_ents
        for texto_ent, label in ambos:
            registros.append({"documento_id": idx, "texto": texto_ent, "tipo": label, "status": "ambos"})
        for texto_ent, label in so_pre:
            registros.append({"documento_id": idx, "texto": texto_ent, "tipo": label, "status": "so_pretreinado"})
        for texto_ent, label in so_cust:
            registros.append({"documento_id": idx, "texto": texto_ent, "tipo": label, "status": "so_customizado"})
    return pd.DataFrame(registros)


def exportar_metrica_ner(metricas: Dict[str, float], caminho: str) -> None:
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as fh:
        json.dump(metricas, fh, ensure_ascii=False, indent=2)
    logger.info("Metricas NER exportadas: %s", caminho)
