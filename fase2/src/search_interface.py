import re

from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)


def _obter_modelo_word2vec(pipeline):
    """Retorna o wv do Word2Vec se disponivel, ou None com mensagem de erro."""
    if "word2vec" not in pipeline.vetorizadores:
        print("[AVISO] Word2Vec nao disponivel. Inclua 'word2vec' em METODOS_EMBEDDING.")
        return None
    modelo = pipeline.vetorizadores["word2vec"].model
    if modelo is None:
        print("[AVISO] Modelo Word2Vec nao foi treinado (dados insuficientes).")
        return None
    return modelo.wv


def iniciar_interface_busca(pipeline):
    methods = list(pipeline.motores_busca.keys())
    if not methods:
        print("[ERRO] Nenhum metodo de busca disponivel.")
        return

    print()
    print("=" * 60)
    print("  SISTEMA DE BUSCA TEXTUAL POR SIMILARIDADE")
    print("=" * 60)
    print(f"  Metodos disponiveis: {', '.join(methods)}")
    print("  Comandos:")
    print("    <consulta>               - busca usando o primeiro metodo")
    print("    <metodo> <consulta>      - busca com metodo especifico")
    print("    similar <palavra>        - palavras similares (Word2Vec)")
    print("    vetor <palavra>          - vetor da palavra (Word2Vec)")
    print("    analog <a> - <b> + <c>  - analogia: a - b + c (Word2Vec)")
    print("    sair                     - encerra")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando...")
            break

        if not user_input:
            continue

        if user_input.lower() == "sair":
            print("Encerrando busca...")
            break

        # --- Comandos Word2Vec ---
        if user_input.lower().startswith("similar "):
            palavra = user_input[8:].strip()
            wv = _obter_modelo_word2vec(pipeline)
            if wv is not None:
                if palavra not in wv:
                    print(f"[AVISO] Palavra '{palavra}' nao esta no vocabulario Word2Vec.")
                else:
                    top_k = pipeline.configuracoes.get("TOP_K_RESULTADOS", 5)
                    resultados = wv.most_similar(palavra, topn=top_k)
                    print(f"\nPalavras mais similares a '{palavra}':")
                    for w, score in resultados:
                        print(f"  {w:<20} {score:.4f}")
            continue

        if user_input.lower().startswith("vetor "):
            palavra = user_input[6:].strip()
            wv = _obter_modelo_word2vec(pipeline)
            if wv is not None:
                if palavra not in wv:
                    print(f"[AVISO] Palavra '{palavra}' nao esta no vocabulario Word2Vec.")
                else:
                    vetor = wv[palavra]
                    print(f"\nVetor de '{palavra}' (dim={len(vetor)}):")
                    print("  " + " ".join(f"{v:.4f}" for v in vetor))
            continue

        m_analog = re.match(r'^analog\s+(.+?)\s+-\s+(.+?)\s+\+\s+(.+)$', user_input, re.IGNORECASE)
        if m_analog:
            a, b, c = m_analog.group(1).strip(), m_analog.group(2).strip(), m_analog.group(3).strip()
            wv = _obter_modelo_word2vec(pipeline)
            if wv is not None:
                ausentes = [p for p in (a, b, c) if p not in wv]
                if ausentes:
                    print(f"[AVISO] Palavras fora do vocabulario: {', '.join(ausentes)}")
                else:
                    top_k = pipeline.configuracoes.get("TOP_K_RESULTADOS", 5)
                    resultados = wv.most_similar(positive=[a, c], negative=[b], topn=top_k)
                    print(f"\nAnalogia: '{a}' - '{b}' + '{c}':")
                    for w, score in resultados:
                        print(f"  {w:<20} {score:.4f}")
            continue

        parts = user_input.split(" ", 1)

        if len(parts) == 2 and parts[0] in methods:
            method = parts[0]
            query = parts[1]
        elif parts[0] in methods:
            method = parts[0]
            query = input("Digite sua consulta: ").strip()
            if not query:
                continue
        else:
            method = methods[0]
            query = user_input

        print(f"\nBuscando com [{method}]: '{query}'")
        print("-" * 50)

        results = pipeline.buscar_texto(method, query, top_k=pipeline.configuracoes.get("TOP_K_RESULTADOS", 10))

        if not results:
            print("  Nenhum resultado encontrado.")
            continue

        for i, r in enumerate(results, 1):
            print(f"  #{i} [Score: {r['score']:.4f}] Doc #{r['index']}")
            print(f"      {r['preview']}")
            print()

        print("-" * 50)
        print(f"  {len(results)} resultados encontrados.")
