# Plano de Processamento de NLP - Wikipedia Articles

## Visão Geral do Projeto

Este projeto processa artigos da Wikipedia em português utilizando spaCy para análise linguística, com foco em pré-processamento, POS tagging, NER e visualização.

## Estrutura do Corpus

**Arquivo de entrada:** `artigos_wikipedia.txt`

**Formato reconhecido:**
```
===== ARTICLE START =====
Title: <título do artigo>
URL: <url do artigo>
=========================
<conteúdo do artigo>
===== ARTICLE END =====
```

## Pipeline de Pré-Processamento

### 1. Carregamento e Inspeção Inicial do Corpus

- Leitura do arquivo `artigos_wikipedia.txt`
- Parsing dos delimitadores `===== ARTICLE START =====` e `===== ARTICLE END =====`
- Extração de metadados (title, url) e conteúdo de cada artigo
- Estatísticas iniciais:
  - Total de artigos
  - Total de tokens/caracteres por artigo
  - Distribuição de tamanhos
- Log: registro de artigos carregados com sucesso e falha

### 2. Tokenização de Palavras e Sentenças

- Utilização do spaCy (`pt_core_news_lg`) para tokenização
- Tokenização de sentenças: `doc.sents`
- Tokenização de palavras: `doc`
- Preservação de tokens, lemmas, e informações morfológicas

### 3. Remoção de Stopwords

- Utilização das stopwords nativas do modelo `pt_core_news_lg`
- Stopwords disponíveis em: `nlp.Defaults.stop_words`
- Adição de stopwords customizadas quando necessário:
  - Palavras frequentes sem informação semântica
  - Números, símbolos especiais
  - Palavras específicas do domínio
- Funcionalidade para adicionar/remover stopwords customizadas

### 4. POS Tagging com spaCy

Implementação conforme exemplo fornecido:

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `artigo_id` | `int` | ID único do artigo |
| `token_id` | `int` | Posição do token no documento |
| `token` | `str` | Texto original do token |
| `pos` | `str` | Classe gramatical universal (Universal POS) |
| `tag` | `str` | Etiqueta detalhada do POS tagger |
| `lemma` | `str` | Lema do token (lemmatizer) |
| `dep_rel` | `str` | Relação de dependência (parser) |
| `head_token` | `str` | Cabeça sintática do token |
| `entity` | `str` | Texto da entidade nomeada (NER) |
| `entity_label` | `str` | Tipo da entidade (NER) |
| `title` | `str` | Título do artigo |
| `url` | `str` | URL original do artigo |

**Parâmetros de processamento:**
- `batch_size`: 10 (modelo lg consome mais RAM)
- Modelo: `pt_core_news_lg`
- Processamento em lote via `nlp.pipe()`

### 5. Nuvem de Palavras (WordCloud)

- Geração a partir do dicionário de frequência de tokens
- Biblioteca: `wordcloud`
- Possibilidades de configuração:
  - Tamanho máximo de palavras
  - Cores (colormap)
  - Background
  - Largura/altura da imagem
- Opções de filtragem:
  - Com/sem stopwords
  - Por POS tag (ex: apenas substantivos)
  - Por artigo específico
- Exportação em PNG

### 6. Análise do Impacto do Pré-Processamento

- Comparação do vocabulário:
  - Antes e depois da remoção de stopwords
  - Quantidade de tokens únicos
  - Palavras mais frequentes (top N)
  - Distribuição de POS tags
- Métricas:
  - Redução percentual do vocabulário
  - Tokens mais impactados
  - Preservação de informações relevantes (NER, etc.)
- Visualizações:
  - Gráfico de frequência de palavras
  - Distribuição de POS tags (barras/pizza)
  - Comparativo antes/depois

### 7. Sistema de Logs

- **Terminal**: saída em tempo real com timestamps
- **Arquivo**: `nlp_pipeline.log` com todo o histórico
- Níveis de log:
  - `INFO`: operações normais
  - `WARNING`: avisos não críticos
  - `ERROR`: falhas que impedem continuidade
- Formato: `[TIMESTAMP] [LEVEL] mensagem`
- Cada etapa do pipeline com log específico

### 8. Testes de Validação

Estrutura de testes em `tests/`:

- **Testes de carregamento**:
  - Leitura correta do arquivo de artigos
  - Parsing dos delimitadores
  - Extração de metadados (title, url)

- **Testes de pré-processamento**:
  - Tokenização correta
  - Remoção de stopwords (padrão + customizadas)
  - Lematização

- **Testes de POS tagging**:
  - Verificação de POS tags não vazios
  - Preservação de entidades
  - Relações de dependência

- **Testes de pipeline**:
  - Integração completa
  - Tratamento de edge cases
  - Performance

### 9. README.md

Documentação com:

- Descrição do projeto
- Requisitos (dependências)
- Instalação
- Como executar o pipeline
- Descrição das features:
  - Carregamento de corpus
  - Pré-processamento
  - POS tagging
  - Nuvem de palavras
  - Análise de vocabulário
- Estrutura de arquivos de saída
- Exemplos de uso
- Testes

## Estrutura de Arquivos do Projeto

```
projeto/
├── artigos_wikipedia.txt    # Entrada: artigos raw
├── plan.md                  # Este plano
├── main.py                  # Script principal
├── src/
│   ├── __init__.py
│   ├── corpus_loader.py    # Carregamento do corpus
│   ├── preprocessing.py    # Pipeline de pré-processamento
│   ├── pos_tagger.py       # POS tagging (spaCy)
│   ├── wordcloud_gen.py    # Geração de wordcloud
│   ├── vocab_analysis.py   # Análise de vocabulário
│   ├── logger.py           # Sistema de logs
│   └── config.py           # Configurações globais
├── tests/
│   ├── __init__.py
│   ├── test_corpus_loader.py
│   ├── test_preprocessing.py
│   ├── test_pos_tagger.py
│   └── test_pipeline.py
├── output/
│   ├── artigos_anotacao_lg.csv   # DataFrame com tokens
│   ├── wordcloud.png             # Nuvem de palavras
│   ├── vocabulario_analise.json # Análise de vocabulário
│   └── nlp_pipeline.log          # Log de execução
├── requirements.txt
└── README.md
```

## Dependências Principais

- `spacy>=3.7.0`
- `pt_core_news_lg` (modelo spaCy)
- `pandas>=2.0.0`
- `wordcloud>=1.9.0`
- `matplotlib>=3.7.0`
- `pytest>=7.0.0`

## Considerações de Implementação

1. **Memória**: O modelo `lg` consome mais RAM, usar `batch_size` reduzido
2. **Encoding**: UTF-8 para compatibilidade com caracteres portugueses
3. **Performance**: Processamento em lote com `nlp.pipe()`
4. **Extensibilidade**: Design modular para permitir variações no pipeline
5. **Reprodutibilidade**: Seeds fixos para random states