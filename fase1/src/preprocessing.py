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
    - Remove URLs (http/https e www)
    - Remove caracteres que não sejam letras (A-Z, À-ÿ), dígitos ou espaços
    - Colapsa múltiplos espaços em um único espaço

    Args:
        texto: Texto bruto a ser normalizado.

    Returns:
        Texto normalizado como string.
    """
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", " ", texto)
    texto = re.sub(r"[^a-zA-Z0-9À-ÿ\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
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


def tokenizar_por_tipo(texto_artigo, tipo_tokenizacao='palavra', metodo_processamento='none'):
    """
    Tokeniza um artigo por palavra.
    O parâmetro tipo_tokenizacao é mantido por compatibilidade, mas nesta etapa
    a tokenização é sempre por palavra.

    Args:
        texto_artigo: Texto do artigo.
        tipo_tokenizacao: ignorado; sempre usa tokenização por palavra.
        metodo_processamento: 'none', 'lemmatizacao' ou 'stemming'.

    Returns:
        Dicionário com 'tokens' e 'sentencas'.
    """
    texto_normalizado = normalizar_texto(texto_artigo)
    nlp = obter_instancia_nlp()
    doc = nlp(texto_normalizado)

    sentencas = [sentenca.text for sentenca in doc.sents]

    tokens = []
    for token in doc:
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
