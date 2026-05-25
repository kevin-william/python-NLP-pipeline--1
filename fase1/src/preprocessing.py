import re
import spacy
from nltk.stem import PorterStemmer, SnowballStemmer
from fase1_config import MODELO_SPACY, STOPWORDS_EXTRAS
from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)

_stopwords_personalizadas = set()
_instancia_nlp = None

_stemmer_porter = PorterStemmer()
_stemmer_snowball = SnowballStemmer(language="portuguese")


def obter_instancia_nlp():
    global _instancia_nlp
    if _instancia_nlp is None:
        logger.info("Carregando modelo spaCy: %s", MODELO_SPACY)
        _instancia_nlp = spacy.load(MODELO_SPACY)
        logger.info("Modelo carregado com sucesso")
    return _instancia_nlp


def normalizar_texto(texto):
    """
    Normaliza o texto antes da tokenização:
    - Converte para letras minúsculas
    - Remove caracteres especiais (mantém letras acentuadas, dígitos e espaços)
    - Remove underscores
    - Colapsa múltiplos espaços em um único espaço

    Args:
        texto: Texto bruto a ser normalizado.

    Returns:
        Texto normalizado como string.
    """
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', ' ', texto)   # remove pontuação e símbolos (mantém acentos via Unicode)
    texto = re.sub(r'_', ' ', texto)           # remove underscore
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto


def aplicar_stemming(texto, metodo='snowball'):
    """
    Aplica stemming ao texto usando o método especificado.

    Args:
        texto: Texto para aplicar stemming.
        metodo: Método de stemming ('porter' ou 'snowball').

    Returns:
        Texto com stemming aplicado.
    """
    if metodo == 'porter':
        return _stemmer_porter.stem(texto)
    return _stemmer_snowball.stem(texto)


def obter_stopwords():
    nlp = obter_instancia_nlp()
    stopwords_extras = set(palavra.lower() for palavra in STOPWORDS_EXTRAS)
    return nlp.Defaults.stop_words | _stopwords_personalizadas | stopwords_extras


def adicionar_stopwords_personalizadas(palavras):
    for palavra in palavras:
        _stopwords_personalizadas.add(palavra.lower())
    logger.info("Stopwords customizadas adicionadas: %s", palavras)


def remover_stopwords_personalizadas(palavras):
    for palavra in palavras:
        _stopwords_personalizadas.discard(palavra.lower())
    logger.info("Stopwords customizadas removidas: %s", palavras)


def extrair_ngramas(tokens, n):
    """
    Constrói n-gramas a partir de uma lista de tokens.

    Args:
        tokens: Lista de tokens (spaCy ou qualquer objeto indexável).
        n: Tamanho do n-grama (2 para bigrama, 3 para trigrama).

    Returns:
        Lista de grupos (sublistas) de tamanho n.
    """
    return [tokens[i:i + n] for i in range(len(tokens) - n + 1)]


def tokenizar_por_tipo(texto_artigo, tipo_tokenizacao='palavra', metodo_processamento='none'):
    """
    Tokeniza um artigo de acordo com o tipo de tokenização especificado.
    Aplica normalização textual antes da tokenização.

    Args:
        texto_artigo: Texto do artigo.
        tipo_tokenizacao: 'palavra', 'bigrama', 'trigrama' ou 'sentenca'.
        metodo_processamento: 'none', 'lemmatizacao' ou 'stemming'.

    Returns:
        Dicionário com 'tokens' e 'sentencas'.
    """
    texto_normalizado = normalizar_texto(texto_artigo)
    nlp = obter_instancia_nlp()
    doc = nlp(texto_normalizado)

    sentencas = [sentenca.text for sentenca in doc.sents]

    if tipo_tokenizacao == 'sentenca':
        tokens = []
        for sentenca in doc.sents:
            lema_sentenca = " ".join(t.lemma_ for t in sentenca)
            if metodo_processamento == 'lemmatizacao':
                processado = lema_sentenca
            elif metodo_processamento == 'stemming':
                processado = " ".join(aplicar_stemming(t.lemma_) for t in sentenca)
            else:
                processado = sentenca.text
            tokens.append({
                "texto": sentenca.text,
                "lema": lema_sentenca,
                "processado": processado,
                "pos": "SENT",
                "tag": "SENT",
                "dependencia": "",
                "eh_stopword": False,
                "eh_pontuacao": False,
                "eh_alfabetico": True,
            })
        return {"tokens": tokens, "sentencas": sentencas}

    tokens_base = list(doc)

    if tipo_tokenizacao == 'palavra':
        tokens = []
        for token in tokens_base:
            lema = token.lemma_
            if metodo_processamento == 'lemmatizacao':
                processado = lema
            elif metodo_processamento == 'stemming':
                processado = aplicar_stemming(lema)
            else:
                processado = token.text
            tokens.append({
                "texto": token.text,
                "lema": lema,
                "processado": processado,
                "pos": token.pos_,
                "tag": token.tag_,
                "dependencia": token.dep_,
                "eh_stopword": token.is_stop,
                "eh_pontuacao": token.is_punct,
                "eh_alfabetico": token.is_alpha,
            })
        return {"tokens": tokens, "sentencas": sentencas}

    # bigrama ou trigrama
    n = 2 if tipo_tokenizacao == 'bigrama' else 3
    grupos = extrair_ngramas(tokens_base, n)
    tokens = []
    for grupo in grupos:
        texto_grupo = " ".join(t.text for t in grupo)
        lema_grupo = " ".join(t.lemma_ for t in grupo)
        if metodo_processamento == 'lemmatizacao':
            processado = lema_grupo
        elif metodo_processamento == 'stemming':
            processado = " ".join(aplicar_stemming(t.lemma_) for t in grupo)
        else:
            processado = texto_grupo
        tokens.append({
            "texto": texto_grupo,
            "lema": lema_grupo,
            "processado": processado,
            "pos": grupo[0].pos_,
            "tag": grupo[0].tag_,
            "dependencia": grupo[0].dep_,
            "eh_stopword": all(t.is_stop for t in grupo),
            "eh_pontuacao": all(t.is_punct for t in grupo),
            "eh_alfabetico": all(t.is_alpha for t in grupo),
        })
    return {"tokens": tokens, "sentencas": sentencas}


def tokenizar_artigo(texto_artigo, metodo_processamento='none'):
    """
    Tokeniza um artigo por palavra com o método de processamento especificado.
    Atalho para tokenizar_por_tipo com tipo_tokenizacao='palavra'.

    Args:
        texto_artigo: Texto do artigo.
        metodo_processamento: Método de processamento ('none', 'lemmatizacao' ou 'stemming').

    Returns:
        Dicionário com 'tokens' e 'sentencas'. Cada token contém o campo 'processado'
        com o valor resultante do método escolhido.
    """
    return tokenizar_por_tipo(texto_artigo, tipo_tokenizacao='palavra', metodo_processamento=metodo_processamento)


def remover_stopwords_dos_tokens(tokens):
    stopwords_atuais = obter_stopwords()
    tokens_filtrados = [
        token for token in tokens
        if token.get("processado", token["texto"]).lower() not in stopwords_atuais
        and not token["eh_pontuacao"]
    ]
    return tokens_filtrados
