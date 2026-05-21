from logger import setup_logger

logger = setup_logger(__name__)


def start_search_interface(pipeline):
    methods = list(pipeline.search_engines.keys())
    if not methods:
        print("[ERRO] Nenhum metodo de busca disponivel.")
        return

    print()
    print("=" * 60)
    print("  SISTEMA DE BUSCA TEXTUAL POR SIMILARIDADE")
    print("=" * 60)
    print(f"  Metodos disponiveis: {', '.join(methods)}")
    print("  Comandos:")
    print("    <consulta>           - busca usando o primeiro metodo")
    print("    <metodo> <consulta>  - busca com metodo especifico")
    print("    sair                 - encerra")
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

        results = pipeline.search_text(method, query, top_k=pipeline.config.get("TOP_K_RESULTS", 10))

        if not results:
            print("  Nenhum resultado encontrado.")
            continue

        for i, r in enumerate(results, 1):
            print(f"  #{i} [Score: {r['score']:.4f}] Doc #{r['index']}")
            print(f"      {r['preview']}")
            print()

        print("-" * 50)
        print(f"  {len(results)} resultados encontrados.")
