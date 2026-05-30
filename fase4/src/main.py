"""
Pipeline principal da Fase 4 — Extracao de Informacoes e Grafo de Conhecimento.

Executa as 11 etapas descritas no plan.md:
1. Inicializacao
2. Carregar dados
3. Extracao de padroes com regex
4. NER com spaCy (modelo pre-treinado)
5. NER customizado
6. Fuzzy matching
7. Extracao de relacoes SVO
8. Construcao do grafo
9. Centralidade e comunidades
10. Pergunta analitica e relatorio
11. Visualizacoes
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict

DIRETORIO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIRETORIO_SCRIPT)
sys.path.insert(0, os.path.join(DIRETORIO_SCRIPT, "..", "..", "shared"))

from fase4_config import (
    CAMINHO_ARTEFATO_FASE2,
    CAMINHO_LOG,
    CAMINHO_MODELO_CUSTOMIZADO,
    CAMINHO_PARQUET_ENTRADA,
    DIRETORIO_DISPLACY,
    DIRETORIO_PLOTS,
    DIRETORIO_SAIDA,
    EPOCAS_TREINO_NER,
    NUMERO_EXEMPLOS_TREINO,
)
from logger import inicializar_sistema_log

# Cria diretorios antes de inicializar logger (que tenta gravar em output/)
os.makedirs(DIRETORIO_SAIDA, exist_ok=True)
os.makedirs(DIRETORIO_PLOTS, exist_ok=True)
os.makedirs(DIRETORIO_DISPLACY, exist_ok=True)
os.makedirs(CAMINHO_MODELO_CUSTOMIZADO, exist_ok=True)

logger = inicializar_sistema_log("fase4")


def _carregar_parquet(caminho: str):
    import pandas as pd

    if not os.path.exists(caminho):
        raise FileNotFoundError(
            f"Parquet da Fase 1 nao encontrado: {caminho}\n"
            "Copie o arquivo para fase4/input/ ou execute a Fase 1 antes."
        )
    return pd.read_parquet(caminho)


def _carregar_artefato(caminho: str):
    """Carrega artefato .lpf2 da Fase 2."""
    if not os.path.exists(caminho):
        return None
    try:
        import joblib

        obj = joblib.load(caminho)
        return obj
    except Exception as e:
        logger.warning("Falha ao carregar artefato Fase 2: %s", e)
        return None


def _reconstruir_documentos_do_parquet(df):
    """Reconstroi textos concatenando tokens por id_artigo."""
    import pandas as pd

    if "id_artigo" not in df.columns:
        col_token = "token" if "token" in df.columns else df.columns[0]
        return [" ".join(df[col_token].dropna().astype(str).tolist())]

    docs = []
    for art_id in sorted(df["id_artigo"].unique()):
        sub = df[df["id_artigo"] == art_id]
        col = "lema" if "lema" in sub.columns else "token"
        tokens = sub[col].dropna().astype(str).tolist()
        docs.append(" ".join(tokens))
    return docs


def executar_fase4_principal() -> Dict[str, Any]:
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("FASE 4 — Extracao de Informacoes e Grafo de Conhecimento")
    logger.info("Inicio: %s", time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 70)

    saidas: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # ETAPA 2: Carregar dados
    # ------------------------------------------------------------------
    logger.info("[Etapa 2] Carregando dados...")
    parquet_df = _carregar_parquet(CAMINHO_PARQUET_ENTRADA)
    logger.info("Parquet carregado: %d linhas, colunas: %s", len(parquet_df), list(parquet_df.columns))

    artefato = _carregar_artefato(CAMINHO_ARTEFATO_FASE2)
    if artefato is not None:
        documentos = artefato.documentos
        titulos = artefato.titulos
        logger.info("Artefato Fase 2 carregado: %d documentos", len(documentos))
    else:
        logger.warning("Artefato Fase 2 ausente — reconstruindo documentos do parquet")
        documentos = _reconstruir_documentos_do_parquet(parquet_df)
        # Tenta obter titulos do parquet
        if "titulo" in parquet_df.columns:
            titulos = parquet_df.groupby("id_artigo")["titulo"].first().tolist() if "id_artigo" in parquet_df.columns else parquet_df["titulo"].drop_duplicates().tolist()
        else:
            titulos = [f"Documento {i}" for i in range(len(documentos))]

    # ------------------------------------------------------------------
    # ETAPA 3: Extracao de padroes com regex
    # ------------------------------------------------------------------
    logger.info("[Etapa 3] Extraindo padroes com regex...")
    from extracao_padroes import extrair_padroes_corpus, gerar_estatisticas_padroes

    df_padroes = extrair_padroes_corpus(documentos, titulos)
    stats_padroes = gerar_estatisticas_padroes(df_padroes)
    logger.info("Padroes extraidos: %d ocorrencias", len(df_padroes))
    logger.info("Estatisticas:\n%s", stats_padroes.to_string(index=False))

    # ------------------------------------------------------------------
    # ETAPA 4: NER com spaCy (modelo pre-treinado)
    # ------------------------------------------------------------------
    logger.info("[Etapa 4] Executando NER com spaCy...")
    from ner_analysis import (
        carregar_modelo_spacy,
        contar_entidades_por_tipo,
        consolidar_entidades_parquet_spacy,
        extrair_entidades_corpus,
        gerar_displacy_corpus,
    )

    nlp = carregar_modelo_spacy()
    df_ents_spacy = extrair_entidades_corpus(documentos, nlp)
    logger.info("Entidades spaCy: %d", len(df_ents_spacy))

    df_ents_consolidado = consolidar_entidades_parquet_spacy(df_ents_spacy, parquet_df)
    logger.info("Entidades consolidadas: %d", len(df_ents_consolidado))
    logger.info("Por tipo:\n%s", contar_entidades_por_tipo(df_ents_consolidado).to_string(index=False))

    # displaCy
    gerar_displacy_corpus(documentos, nlp, DIRETORIO_DISPLACY)

    # Salva entidades extraidas
    caminho_ents = os.path.join(DIRETORIO_SAIDA, "entidades_extraidas.csv")
    df_ents_consolidado.to_csv(caminho_ents, index=False, encoding="utf-8")
    saidas["entidades_extraidas"] = caminho_ents
    logger.info("Entidades salvas: %s", caminho_ents)

    # ------------------------------------------------------------------
    # ETAPA 5: NER customizado
    # ------------------------------------------------------------------
    logger.info("[Etapa 5] Treinando NER customizado...")
    from ner_customizado import (
        avaliar_modelo_ner,
        comparar_modelos,
        converter_entidades_para_formato_treino,
        exportar_metrica_ner,
        treinar_ner_customizado,
    )

    dados_treino = converter_entidades_para_formato_treino(parquet_df, documentos)
    n_treino = int(len(dados_treino) * 0.8)
    dados_teste = dados_treino[n_treino:] if len(dados_treino) > 5 else dados_treino[:5]
    dados_treino_real = dados_treino[:n_treino] if len(dados_treino) > 5 else dados_treino

    modelo_custom = treinar_ner_customizado(dados_treino_real, epochs=EPOCAS_TREINO_NER)
    metricas_ner = avaliar_modelo_ner(modelo_custom, dados_teste)
    logger.info("Metricas NER customizado: %s", metricas_ner)

    caminho_metricas_ner = os.path.join(CAMINHO_MODELO_CUSTOMIZADO, "metricas_avaliacao.json")
    exportar_metrica_ner(metricas_ner, caminho_metricas_ner)

    df_comparacao_ner = comparar_modelos(nlp, modelo_custom, documentos[:20])
    caminho_comp_ner = os.path.join(CAMINHO_MODELO_CUSTOMIZADO, "comparacao_modelos.csv")
    df_comparacao_ner.to_csv(caminho_comp_ner, index=False, encoding="utf-8")
    saidas["metricas_ner"] = caminho_metricas_ner
    saidas["comparacao_ner"] = caminho_comp_ner

    # ------------------------------------------------------------------
    # ETAPA 6: Fuzzy matching
    # ------------------------------------------------------------------
    logger.info("[Etapa 6] Aplicando fuzzy matching (Levenshtein)...")
    from fuzzy_matching import normalizar_entidades

    entidades_unicas = df_ents_consolidado["texto"].dropna().unique().tolist()
    logger.info("Entidades unicas para fuzzy: %d", len(entidades_unicas))

    df_fuzzy = normalizar_entidades(entidades_unicas)
    caminho_fuzzy = os.path.join(DIRETORIO_SAIDA, "fuzzy_matches.csv")
    df_fuzzy.to_csv(caminho_fuzzy, index=False, encoding="utf-8")
    saidas["fuzzy_matches"] = caminho_fuzzy
    logger.info("Fuzzy matches: %d grupos", df_fuzzy["grupo_id"].nunique())

    # Aplica canonicalizacao: substituir variantes por forma canonica
    mapa_canonica = dict(
        zip(df_fuzzy["entidade_original"], df_fuzzy["entidade_canonica"])
    )
    df_ents_norm = df_ents_consolidado.copy()
    df_ents_norm["texto"] = df_ents_norm["texto"].map(
        lambda x: mapa_canonica.get(x, x)
    )

    # ------------------------------------------------------------------
    # ETAPA 7: Extracao de relacoes SVO
    # ------------------------------------------------------------------
    logger.info("[Etapa 7] Extraindo relacoes SVO...")
    from relacoes import (
        construir_arestas_grafo,
        extrair_relacoes_corpus,
        filtrar_relacoes_relevantes,
    )

    df_relacoes = extrair_relacoes_corpus(documentos, nlp)
    logger.info("Relacoes extraidas: %d", len(df_relacoes))

    df_arestas = construir_arestas_grafo(df_relacoes)
    df_arestas_filtradas = filtrar_relacoes_relevantes(df_arestas, min_freq=1)

    caminho_relacoes = os.path.join(DIRETORIO_SAIDA, "relacoes_extraidas.csv")
    df_relacoes.to_csv(caminho_relacoes, index=False, encoding="utf-8")
    saidas["relacoes_extraidas"] = caminho_relacoes
    logger.info("Relacoes salvas: %s", caminho_relacoes)

    # ------------------------------------------------------------------
    # ETAPA 8: Construcao do grafo
    # ------------------------------------------------------------------
    logger.info("[Etapa 8] Construindo grafo de conhecimento...")
    from grafo_conhecimento import GrafoConhecimento

    gc = GrafoConhecimento()
    gc.construir_grafo_entidade_entidade(df_arestas_filtradas)
    gc.adicionar_nos_entidades(df_ents_norm)

    logger.info(
        "Grafo: %d nos | %d arestas",
        gc.grafo.number_of_nodes(),
        gc.grafo.number_of_edges(),
    )

    # ------------------------------------------------------------------
    # ETAPA 9: Centralidade e comunidades
    # ------------------------------------------------------------------
    logger.info("[Etapa 9] Calculando centralidade e detectando comunidades...")
    centralidade = gc.calcular_centralidade()
    comunidades = gc.detectar_comunidades()
    logger.info("Comunidades: %d", len(comunidades))

    ranking = gc.obter_ranking_consolidado()
    logger.info("Top entidades (ranking consolidado):\n%s", ranking.to_string(index=False))

    arestas_freq = gc.obter_arestas_mais_frequentes()

    # ------------------------------------------------------------------
    # ETAPA 10: Pergunta analitica e relatorio interpretativo
    # ------------------------------------------------------------------
    logger.info("[Etapa 10] Gerando relatorio interpretativo...")
    resposta = gc.responder_pergunta_analitica()
    resumo = gc.gerar_resumo_interpretativo(centralidade, arestas_freq)

    relatorio = "\n\n".join([resposta, resumo])
    caminho_relatorio = os.path.join(DIRETORIO_SAIDA, "relatorio_interpretativo.txt")
    with open(caminho_relatorio, "w", encoding="utf-8") as fh:
        fh.write(relatorio)
    saidas["relatorio_interpretativo"] = caminho_relatorio
    logger.info("Relatorio salvo: %s", caminho_relatorio)
    logger.info("\n%s", resposta)

    # ------------------------------------------------------------------
    # ETAPA 11: Visualizacoes
    # ------------------------------------------------------------------
    logger.info("[Etapa 11] Gerando visualizacoes...")
    from visualizacao_grafo import (
        gerar_infografico_resumo,
        plotar_comparacao_centralidades,
        plotar_comunidades_grafo,
        plotar_distribuicao_centralidade,
        plotar_frequencia_entidades,
        plotar_grafo_matplotlib,
        plotar_grafo_pyvis,
        plotar_heatmap_relacoes,
        plotar_top_entidades_por_tipo,
    )

    # Grafo estatico
    caminho_grafo_png = os.path.join(DIRETORIO_PLOTS, "grafo_conhecimento.png")
    plotar_grafo_matplotlib(gc.grafo, centralidade, caminho_grafo_png)
    saidas["grafo_conhecimento_png"] = caminho_grafo_png

    # Grafo interativo
    caminho_grafo_html = os.path.join(DIRETORIO_PLOTS, "grafo_interativo.html")
    plotar_grafo_pyvis(gc.grafo, centralidade, caminho_grafo_html)
    saidas["grafo_interativo_html"] = caminho_grafo_html

    # Distribuicao de centralidade
    caminho_centralidade = os.path.join(DIRETORIO_PLOTS, "centralidade_entidades.png")
    df_bet = centralidade.get("betweenness", centralidade.get("degree", None))
    if df_bet is not None and not df_bet.empty:
        plotar_distribuicao_centralidade(df_bet, "betweenness", caminho_centralidade)
    saidas["centralidade_png"] = caminho_centralidade

    # Comparacao de centralidades
    caminho_comp = os.path.join(DIRETORIO_PLOTS, "comparacao_centralidades.png")
    plotar_comparacao_centralidades(centralidade, caminho_comp)
    saidas["comparacao_centralidades_png"] = caminho_comp

    # Top entidades por tipo
    caminho_top_tipos = os.path.join(DIRETORIO_PLOTS, "top_entidades_por_tipo.png")
    plotar_top_entidades_por_tipo(df_ents_norm, caminho_top_tipos)
    saidas["top_entidades_tipo_png"] = caminho_top_tipos

    # Frequencia
    caminho_freq = os.path.join(DIRETORIO_PLOTS, "frequencia_entidades.png")
    plotar_frequencia_entidades(df_ents_norm, caminho_freq)
    saidas["frequencia_entidades_png"] = caminho_freq

    # Heatmap
    caminho_heatmap = os.path.join(DIRETORIO_PLOTS, "heatmap_relacoes.png")
    plotar_heatmap_relacoes(df_arestas_filtradas, caminho_heatmap)
    saidas["heatmap_relacoes_png"] = caminho_heatmap

    # Comunidades
    caminho_comunidades = os.path.join(DIRETORIO_PLOTS, "comunidades_grafo.png")
    plotar_comunidades_grafo(gc.grafo, comunidades, caminho_comunidades)
    saidas["comunidades_grafo_png"] = caminho_comunidades

    # Infografico resumo
    caminho_infografico = os.path.join(DIRETORIO_PLOTS, "infografico_resumo.png")
    gerar_infografico_resumo(gc.grafo, centralidade, df_ents_norm, resposta, caminho_infografico)
    saidas["infografico_resumo_png"] = caminho_infografico

    # Exporta CSVs do grafo
    caminho_edges_csv = os.path.join(DIRETORIO_SAIDA, "grafo_edges.csv")
    caminho_nos_csv = os.path.join(DIRETORIO_SAIDA, "nos_grafo.csv")
    gc.exportar_grafo(caminho_edges_csv, caminho_nos_csv)
    saidas["grafo_edges_csv"] = caminho_edges_csv
    saidas["nos_grafo_csv"] = caminho_nos_csv

    # Resumo de metricas JSON
    import networkx as nx

    metricas_resumo = {
        "nos": gc.grafo.number_of_nodes(),
        "arestas": gc.grafo.number_of_edges(),
        "densidade": nx.density(gc.grafo) if gc.grafo.number_of_nodes() > 1 else 0.0,
        "componentes": nx.number_connected_components(gc.grafo),
        "comunidades": len(comunidades),
        "total_entidades_extraidas": len(df_ents_consolidado),
        "total_relacoes_extraidas": len(df_relacoes),
        "total_padroes_regex": len(df_padroes),
        "metricas_ner": metricas_ner,
    }
    if gc.grafo.number_of_nodes() > 0:
        try:
            if nx.is_connected(gc.grafo):
                metricas_resumo["diametro"] = nx.diameter(gc.grafo)
                metricas_resumo["clustering_medio"] = nx.average_clustering(gc.grafo)
        except Exception:
            pass

    caminho_metricas_json = os.path.join(DIRETORIO_SAIDA, "resumo_metricas.json")
    with open(caminho_metricas_json, "w", encoding="utf-8") as fh:
        json.dump(metricas_resumo, fh, ensure_ascii=False, indent=2)
    saidas["resumo_metricas_json"] = caminho_metricas_json

    # ------------------------------------------------------------------
    # Finalizacao
    # ------------------------------------------------------------------
    tempo_total = time.time() - t0
    logger.info("=" * 70)
    logger.info("FASE 4 CONCLUIDA em %.1f segundos", tempo_total)
    logger.info("Saidas geradas:")
    for nome, caminho in saidas.items():
        logger.info("  %-40s %s", nome, caminho)
    logger.info("=" * 70)

    return saidas


if __name__ == "__main__":
    executar_fase4_principal()
