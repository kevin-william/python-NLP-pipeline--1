# Plano: Configuração Externa para t-SNE

## Objetivo
Tornar os parâmetros do t-SNE configuráveis externamente para melhorar a visualização dos embeddings.

## Estrutura Atual
O código já suporta configuração via dicionário (linha 120-125 em `embedding_pipeline.py`):
```python
tsne_params = self.config.get("TSNE_PARAMS", {})
visualizer = TSNEVisualizer(**tsne_params)
```

Os parâmetros são definidos em `fase2_config.py`.

---

## Opções de Implementação

### Opção A - Editar `fase2_config.py` (Recomendada - mais simples)
Modificar o dicionário `TSNE_PARAMS` existente.

### Opção B - Criar arquivo YAML/JSON separado
Criar `tsne_config.yaml` e modificar código para ler o arquivo.

---

## Parâmetros do t-SNE

| Parâmetro | Valor Atual | Sugerido | Descrição |
|-----------|-------------|----------|-----------|
| `perplexity` | 5 | 30-50 | Vizinhos considerados. Baixo = clusters compactos. Alto = distribuição uniforme |
| `n_iter` | 1000 | 2000-3000 | Iterações de otimização |
| `n_components` | 2 | 2 | Dimensões do resultado (2D ou 3D) |
| `random_state` | 42 | 42 | Semente para reprodutibilidade |

### Parâmetros do Plot (visualização)

| Parâmetro | Valor Atual | Sugerido | Descrição |
|-----------|-------------|----------|-----------|
| `figsize` | (12, 10) | (16, 12) | Tamanho da figura em polegadas |
| `dpi` | 150 | 150 | Resolução |
| `marker_size` | - | 50 | Tamanho dos pontos |

---

## Passos de Implementação

### Passo 1: Modificar `fase2_config.py`
- Atualizar `TSNE_PARAMS` com valores sugeridos
- Adicionar novos parâmetros de plot

### Passo 2: Atualizar `tsne_plot.py` (se necessário)
- Modificar método `plot()` para aceitar parâmetros de customização

### Passo 3: Executar testes
```bash
cd fase2/src
python -m pytest ../tests/ -v
```
- Verificar se todos os testes passam
- Validar que a visualização foi melhorada

### Passo 4: Atualizar documentação
- Adicionar seção no `README.md` explicando os parâmetros configuráveis
- Incluir exemplos de valores recomendados para diferentes cenários

---

## Valores Recomendados por Cenário

| Cenário | perplexity | n_iter | figsize |
|---------|------------|--------|---------|
| Poucos dados (<100) | 5-10 | 1000 | (10, 8) |
| Dados médios (100-1000) | 30-50 | 2000 | (14, 10) |
| Muitos dados (>1000) | 50-100 | 3000 | (16, 12) |

---

## Critérios de Validação
- [ ] Testes unitários passam
- [ ] Testes de integração passam
- [ ] Visualização t-SNE não está mais condensada
- [ ] README atualizado com novos parâmetros