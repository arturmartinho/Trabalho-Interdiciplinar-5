# Cronograma

---

## [Semana 1]

> **Configuração do ambiente de desenvolvimento** #desenvolvimento
> Instalação e configuração do Ollama nas máquinas de teste, verificação de compatibilidade de hardware (CPU/GPU/RAM) e download dos modelos leves selecionados (Phi-3.5-mini, Qwen2.5-Coder-0.5B, AILO-152M-v2).
>
> - [Ollama (Site Oficial)](https://ollama.com)
> - [Ollama API (Documentação)](https://github.com/ollama/ollama/blob/main/docs/api.md)

> **Levantamento bibliográfico inicial** #pesquisa
> Revisão sistemática da literatura sobre serving de LLMs, load balancing e roteamento de requisições. Organização das referências base (Kossmann2025, Sun2024, Xia2025, Jain2025, Yuan2025).
>
> - [Performance Aware LLM Load Balancer (Paper)](https://doi.org/10.1145/3721146.3721947)
> - [SkyWalker: Locality-Aware Cross-Region Load Balancer](https://doi.org/10.1145/3767295.3769353)

> **Definição do escopo e arquitetura do sistema** #pesquisa #desenvolvimento
> Delimitação dos objetivos do artigo, definição da arquitetura do LoadBalancer (classificador de prompt + proxy para múltiplos modelos) e escolha das métricas de avaliação (latência média, P95, uso de RAM/GPU).
>
> - [Llumnix: Dynamic Scheduling for LLM Serving](https://github.com/AlibabaPAI/llumnix)

---

## [Semana 2]

> **Desenvolvimento do classificador de prompts** #desenvolvimento
> Implementação do classificador por regras e/ou embeddings + classificador leve (Logistic Regression) para detectar intenções como `coding`, `simple_chat`, `math`, etc.
>
> - [Hugging Face](https://huggingface.co)
> - [FastAPI (Documentação)](https://fastapi.tiangolo.com)

> **Implementação do LoadBalancer (versão inicial)** #desenvolvimento
> Criação do serviço em FastAPI com endpoint `/v1/chat/completions`, integração com httpx para consulta às máquinas via `/v1/models` e roteamento básico por regras.
>
> - [Ollama API (Documentação)](https://github.com/ollama/ollama/blob/main/docs/api.md)
> - [httpx (Documentação)](https://www.python-httpx.org)

> **Escrita da fundamentação teórica** #pesquisa
> Redação da seção de fundamentação teórica do artigo, abordando LLMs, serving, heterogeneidade de requisições e estratégias de balanceamento.
>
> - [Llumnix: Dynamic Scheduling for LLM Serving](https://doi.org/10.1145/3721146.3721947)
> - [SkyWalker: Locality-Aware Cross-Region Load Balancer](https://doi.org/10.1145/3767295.3769353)

---

## [Semana 3] — Sprint 2

> **Detalhamento da implementação** #desenvolvimento #pesquisa
> Documentação técnica da arquitetura implementada: diagramas de fluxo, decisões de projeto, integração entre classificador e LoadBalancer, e descrição do mecanismo de consulta aos modelos disponíveis.
>
> - [FastAPI (Documentação)](https://fastapi.tiangolo.com)
> - [Ollama API (Documentação)](https://github.com/ollama/ollama/blob/main/docs/api.md)

> **Coleta de resultados preliminares** #desenvolvimento #pesquisa
> Execução de um conjunto inicial de prompts com o Baseline (modelo maior) e com a Proposta (LoadBalancer), medindo latência média, P95 e uso de recursos. Geração de gráficos e tabelas iniciais.
>
> - [Performance Aware LLM Load Balancer (Paper)](https://doi.org/10.1145/3721146.3721947)

> **Preparação da apresentação oral do Sprint 2** #apresentação
> Criação de slides com arquitetura, resultados preliminares e próximos passos. Ensaios da apresentação oral.
>
> - [Llumnix: Dynamic Scheduling for LLM Serving](https://github.com/AlibabaPAI/llumnix)

> **Apresentação oral do Sprint 2** #apresentação
> Apresentação do progresso: implementação, resultados preliminares e plano para as próximas semanas.
>
> - [Performance Aware LLM Load Balancer (Paper)](https://doi.org/10.1145/3721146.3721947)

---

## [Semana 4]

> **Refinamento do classificador de prompts** #desenvolvimento
> Ajuste do classificador com base nos resultados preliminares. Inclusão de novas categorias de intenção e melhoria da acurácia.
>
> - [Hugging Face](https://huggingface.co)

> **Implementação de métricas de monitoramento** #desenvolvimento
> Integração de logs com timestamp de entrada/saída, coleta de latência com `time.perf_counter()` e monitoramento de uso de RAM/GPU por máquina.
>
> - [Performance Aware LLM Load Balancer (Paper)](https://doi.org/10.1145/3721146.3721947)

> **Escrita da seção de metodologia** #pesquisa
> Redação da metodologia do artigo: descrição do ambiente experimental, modelos utilizados, conjunto de prompts e métricas de avaliação.
>
> - [SkyWalker: Locality-Aware Cross-Region Load Balancer](https://doi.org/10.1145/3767295.3769353)

---

## [Semana 5]

> **Expansão do conjunto de testes** #desenvolvimento #pesquisa
> Criação de um conjunto diversificado de prompts cobrindo diferentes tarefas (coding, chat, math, tradução, etc.) para avaliação robusta.
>
> - [Hugging Face](https://huggingface.co)

> **Execução de experimentos comparativos** #desenvolvimento #pesquisa
> Execução do conjunto de testes com Baseline e Proposta. Coleta sistemática de latência média, P95, P99 e uso de recursos.
>
> - [Performance Aware LLM Load Balancer (Paper)](https://doi.org/10.1145/3721146.3721947)

> **Análise estatística dos resultados** #pesquisa
> Tratamento estatístico dos dados coletados, identificação de outliers e geração de gráficos comparativos.
>
> - [pytest (Documentação)](https://docs.pytest.org)

---

## [Semana 6]

> **Otimização do LoadBalancer** #desenvolvimento
> Melhoria do mecanismo de roteamento com base nos resultados experimentais. Implementação de estratégias alternativas (ex: least loaded, cache affinity).
>
> - [Llumnix: Dynamic Scheduling for LLM Serving](https://github.com/AlibabaPAI/llumnix)
> - [SkyWalker: Locality-Aware Cross-Region Load Balancer](https://doi.org/10.1145/3767295.3769353)

> **Testes automatizados de roteamento** #desenvolvimento
> Criação de testes com pytest para validar o comportamento do LoadBalancer em diferentes cenários.
>
> - [pytest (Documentação)](https://docs.pytest.org)

> **Escrita da seção de resultados** #pesquisa
> Redação da seção de resultados do artigo com base nos dados coletados e análises realizadas.
>
> - [Performance Aware LLM Load Balancer (Paper)](https://doi.org/10.1145/3721146.3721947)

---

## [Semana 7]

> **Comparação com trabalhos relacionados** #pesquisa
> Análise comparativa detalhada entre a abordagem proposta e trabalhos como Llumnix, SkyWalker, DualMap, Preble e Mooncake. Identificação de diferenças, vantagens e limitações.
>
> - [Llumnix: Dynamic Scheduling for LLM Serving](https://github.com/AlibabaPAI/llumnix)
> - [SkyWalker: Locality-Aware Cross-Region Load Balancer](https://doi.org/10.1145/3767295.3769353)
> - [DualMap: Enabling Both Cache Affinity and Load Balancing](https://github.com/ASISys/DualMap)

> **Redação da seção de comparação** #pesquisa
> Escrita da seção do artigo que posiciona a contribuição em relação ao estado da arte.
>
> - [Performance Aware LLM Load Balancer (Paper)](https://doi.org/10.1145/3721146.3721947)

> **Preparação de material visual para o pitch** #apresentação
> Criação de gráficos, diagramas e elementos visuais que serão usados no vídeo de pitch comercial.
>
> - [SkyWalker: Locality-Aware Cross-Region Load Balancer](https://doi.org/10.1145/3767295.3769353)

---

## [Semana 8]

> **Redação das conclusões** #pesquisa
> Escrita da seção de conclusão do artigo, sintetizando contribuições, resultados e trabalhos futuros.
>
> - [Llumnix: Dynamic Scheduling for LLM Serving](https://github.com/AlibabaPAI/llumnix)

> **Revisão e refinamento do artigo** #pesquisa
> Revisão completa do texto, verificação de referências, ajustes de formatação e melhoria da clareza.
>
> - [Performance Aware LLM Load Balancer (Paper)](https://doi.org/10.1145/3721146.3721947)

> **Roteiro do pitch comercial** #apresentação
> Desenvolvimento do roteiro para o vídeo de 2 a 3 minutos, destacando o problema, a solução proposta, os resultados e o valor comercial.
>
> - [SkyWalker: Locality-Aware Cross-Region Load Balancer](https://doi.org/10.1145/3767295.3769353)

---

## [Semana 9] — Sprint 3

> **Gravação e edição do pitch comercial** #apresentação
> Produção do vídeo de 2 a 3 minutos com base no roteiro desenvolvido. Edição, narração e inclusão de elementos visuais.
>
> - [Performance Aware LLM Load Balancer (Paper)](https://doi.org/10.1145/3721146.3721947)

> **Preparação da apresentação oral final** #apresentação
> Criação de slides finais com comparação com trabalhos relacionados, conclusões e resultados consolidados. Ensaios da apresentação.
>
> - [Llumnix: Dynamic Scheduling for LLM Serving](https://github.com/AlibabaPAI/llumnix)
> - [SkyWalker: Locality-Aware Cross-Region Load Balancer](https://doi.org/10.1145/3767295.3769353)

> **Apresentação oral do Sprint 3 e pitch comercial** #apresentação
> Apresentação final do projeto: comparação com trabalhos relacionados, conclusões, pitch comercial e resultados consolidados.
>
> - [DualMap: Enabling Both Cache Affinity and Load Balancing](https://github.com/ASISys/DualMap)

---

## [Semana 10]

> **Submissão final do artigo** #pesquisa #apresentação
> Revisão final do artigo, verificação de formatação e submissão. Entrega do vídeo de pitch comercial e materiais de apresentação.
>
> - [Performance Aware LLM Load Balancer (Paper)](https://doi.org/10.1145/3721146.3721947)
> - [SkyWalker: Locality-Aware Cross-Region Load Balancer](https://doi.org/10.1145/3767295.3769353)

> **Documentação do projeto** #desenvolvimento
> Finalização da documentação técnica do código, README do repositório e instruções de reprodução dos experimentos.
>
> - [Ollama (Site Oficial)](https://ollama.com)
> - [FastAPI (Documentação)](https://fastapi.tiangolo.com)

---
