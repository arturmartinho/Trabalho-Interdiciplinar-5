# Avaliação do classificador de prompts

> Gerado por `experiments/avaliar_classificador.py`.

- Tokenizador: `microsoft/Phi-3.5-mini-instruct` | limites: 50/240 tokens | limiar de confiança do híbrido: 0.55
- Teste: 208 prompts gerados (split de teste) | manuais: 40 prompts escritos à mão
- *Subdimensionado* = enviado a um modelo menor que o necessário (risco de qualidade); *superdimensionado* = a um maior (custa latência).

## Conjunto: teste

| Estratégia | n | Acurácia | F1-macro | Subdim. | Superdim. | Tempo médio (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| regras_palavras | 208 | 79.8% | 0.783 | 8.6% | 11.5% | 0.22 | 0.57 |
| regras | 208 | 78.4% | 0.768 | 12.0% | 9.6% | 4.03 | 0.64 |
| ml_embed | 208 | 98.1% | 0.979 | 1.4% | 0.5% | 439.08 | 570.00 |
| hibrido_embed | 208 | 96.2% | 0.957 | 2.4% | 1.4% | 1.56 | 2.70 |
| ml_tfidf | 208 | 94.2% | 0.936 | 2.9% | 2.9% | 2.34 | 3.07 |
| hibrido_tfidf | 208 | 94.2% | 0.934 | 3.4% | 2.4% | 5.44 | 8.62 |
| llm_gemma4:e4b-it-qat | 60 | 86.7% | 0.823 | 1.7% | 11.7% | 4107.96 | 6343.92 |

> Nota: nesta rodada o tempo de `hibrido_embed` (~1,5 ms) está subestimado. Ele reaproveitou o cache de embeddings de `ml_embed`, que rodou antes; o custo real é o mesmo de `ml_embed` (~450 ms por prompt, uma chamada ao Ollama), exceto quando há sintaxe de código. O script já foi corrigido e limpa o cache entre estratégias.

## Conjunto: manuais

| Estratégia | n | Acurácia | F1-macro | Subdim. | Superdim. | Tempo médio (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| regras_palavras | 40 | 100.0% | 1.000 | 0.0% | 0.0% | 0.28 | 0.62 |
| regras | 40 | 100.0% | 1.000 | 0.0% | 0.0% | 0.34 | 0.68 |
| ml_embed | 40 | 97.5% | 0.974 | 2.5% | 0.0% | 483.01 | 622.91 |
| hibrido_embed | 40 | 100.0% | 1.000 | 0.0% | 0.0% | 1.35 | 1.93 |
| ml_tfidf | 40 | 97.5% | 0.974 | 2.5% | 0.0% | 2.69 | 4.22 |
| hibrido_tfidf | 40 | 100.0% | 1.000 | 0.0% | 0.0% | 6.20 | 7.82 |
| llm_gemma4:e4b-it-qat | 40 | 97.5% | 0.974 | 0.0% | 2.5% | 4890.51 | 7610.74 |

## Detalhe da melhor estratégia no teste: `ml_embed`

Matriz de confusão (linhas = esperado, colunas = previsto):

| | simples | codigo | complexo |
|---|---:|---:|---:|
| **simples** | 89 | 1 | 0 |
| **codigo** | 0 | 65 | 0 |
| **complexo** | 0 | 3 | 50 |

Acurácia por idioma, regras x `ml_embed`:

| idioma | regras | ml_embed |
|---|---:|---:|
| en | 77.1% | 98.6% |
| es | 77.9% | 98.5% |
| pt | 80.0% | 97.1% |

Acurácia por subtipo, regras x `ml_embed`:

| subtipo | regras | ml_embed |
|---|---:|---:|
| argumentacao | 50.0% | 100.0% |
| aritmetica | 100.0% | 100.0% |
| comparacao | 66.7% | 100.0% |
| consulta_sql_regex | 100.0% | 100.0% |
| conversa | 100.0% | 100.0% |
| conversao | 100.0% | 100.0% |
| curto_dificil | 25.0% | 100.0% |
| debug | 100.0% | 100.0% |
| definicao | 88.9% | 100.0% |
| escrever_funcao | 93.3% | 100.0% |
| explicacao_profunda | 66.7% | 83.3% |
| explicar_codigo | 100.0% | 100.0% |
| factual | 100.0% | 100.0% |
| ferramentas | 80.0% | 100.0% |
| gramatica | 88.9% | 100.0% |
| logica | 83.3% | 83.3% |
| multietapas | 100.0% | 100.0% |
| planejamento | 66.7% | 83.3% |
| prova | 66.7% | 100.0% |
| reescrita | 100.0% | 100.0% |
| refatorar | 83.3% | 100.0% |
| resumo_texto | 100.0% | 100.0% |
| sem_palavra_chave | 13.3% | 100.0% |
| sentimento | 100.0% | 88.9% |
| termo_tecnico | 0.0% | 100.0% |
| testes | 100.0% | 100.0% |
| traducao | 100.0% | 100.0% |

Acurácia por caso difícil (adversarial), regras x `ml_embed`:

| caso difícil (adversarial) | regras | ml_embed |
|---|---:|---:|
| False | 90.3% | 97.7% |
| True | 12.5% | 100.0% |

Acurácia por linguagem de programação, regras x `ml_embed`:

| linguagem de programação | regras | ml_embed |
|---|---:|---:|
| Bash | 100.0% | 100.0% |
| C | 80.0% | 100.0% |
| C# | 100.0% | 100.0% |
| Go | 100.0% | 100.0% |
| Java | 100.0% | 100.0% |
| JavaScript | 100.0% | 100.0% |
| PHP | 80.0% | 100.0% |
| Python | 100.0% | 100.0% |
| Rust | 100.0% | 100.0% |
| SQL | 100.0% | 100.0% |
| TypeScript | 100.0% | 100.0% |

## Exemplos de erro de `ml_embed` (teste)

- `GX-2743cf` esperado **complexo**, previsto **codigo**: Si estoy trabajando con Python, ¿podrías explicarme el concepto de 'métodos mágicos' o 'dunder methods' en POO y darme un ejemplo funcional de `__str__` y `__repr__`?
- `GX-ca5ebc` esperado **complexo**, previsto **codigo**: Meu código, que simula um sistema de roteamento de pacotes, está gerando um loop infinito sob certas condições. O bloco de controle é este: `while pacote_pendente: if status == 'EM
- `GX-5b3e55` esperado **complexo**, previsto **codigo**: Pode elaborar um cronograma de estudos para eu aprender Python do zero e conseguir criar um pequeno bot de automação? Tenho 4 meses para dominar o básico e eu quero conseguir imple
- `GS-41b849` esperado **simples**, previsto **codigo**: I need help figuring out this login error: 'Authentication failed for user_id: 123'.

## Limitações

- O rótulo dos prompts gerados é a classe pedida ao LLM gerador, não uma anotação humana; revise uma amostra antes de usar os números no artigo.
- Treino e teste vêm do mesmo gerador e do mesmo template: a acurácia no teste tende a ser otimista. O conjunto manual é a medida fora da distribuição.
