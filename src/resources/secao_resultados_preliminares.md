# Resultados Preliminares

Esta seção apresenta uma primeira comparação entre os dois cenários descritos na metodologia. Os valores são preliminares e servem para verificar se o experimento é executável e se a diferença entre os cenários é perceptível. A análise estatística completa, com tratamento de *outliers* e repetições adicionais, fica para a semana 5 do cronograma.

**Conjunto de prompts.** Foram usados 40 prompts com identificadores fixos, divididos em 14 simples (tarefas que qualquer modelo costuma completar), 13 de código com dificuldade crescente (de uma função trivial a um compilador com máquina virtual) e 13 complexos (problemas difíceis ou custosos, com resposta verificável). O mesmo conjunto foi executado nos dois cenários. Cada prompt foi medido [N] vez(es) por cenário, resultando em [N × 40] amostras por cenário.

**Cenários.** No *baseline*, toda requisição foi enviada diretamente ao Phi-3.5-mini no Ollama. Na *proposta*, a requisição passou pelo endpoint `/infer` do *load balancer*, que classificou o prompt e o encaminhou ao modelo escolhido, com redirecionamento ao Phi-3.5-mini em caso de falha. A latência é o tempo de ponta a ponta medido com `time.perf_counter()` no cliente. Em cada rodada, cada prompt foi executado uma vez em cada cenário, em dois passos com ordem embaralhada por semente fixa. Quem executava primeiro foi balanceado dentro de cada classe, e o mesmo prompt nunca foi executado em sequência, para que o cache de prompt do Ollama não favorecesse o cenário executado depois. O limite de tokens por resposta foi o mesmo nos dois cenários ([N] tokens).

**Aquecimento.** [Houve / Não houve] aquecimento antes da medição: [duas] passadas com um prompt de cada classe, em ambos os cenários, descartadas do registro, para que os três modelos já estivessem carregados na memória. *[CONFIRMAR após a rodada real.]*

**Resultados.** A Tabela 1 e as Figuras 1 e 2 resumem as latências.

**Tabela 1.** Latência (s) por cenário, amostra preliminar.

| Cenário  |   n | Média | P95 | P99 |
|----------|----:|------:|----:|----:|
| Baseline | [ ] |   [ ] | [ ] | [ ] |
| Proposta | [ ] |   [ ] | [ ] | [ ] |

![Latência média, P95 e P99 nos dois cenários](graficos/grafico1_latencia_geral.png)

*Figura 1. Latência média, P95 e P99 nos dois cenários.*

![Latência média por classe de prompt](graficos/grafico2_latencia_por_classe.png)

*Figura 2. Latência média por classe de prompt.*

[Comentar aqui apenas o que os números mostram: variação da média, comportamento do P95/P99 e por qual classe a diferença vem.]

**Limitações.** A amostra é pequena e os percentis P95 e P99 dependem de poucas observações, então não sustentam conclusões estatísticas. O experimento foi executado em uma única máquina local, descrita na subseção de ambiente experimental, e mede apenas latência; não avalia a qualidade das respostas dos modelos menores.
