# TI5 - LLM LoadBalancer

- Artur Fernandes Braga de Menezes
- Artur Rizzi Martinho
- Felipe Lacerda Tertuliano
- Gabriel Alves da Silva Diógenes
- Rafael Mortimer Colares

Esta branch entrega o núcleo do load balancer. A API recebe o prompt, chama o classificador, envia o texto ao modelo escolhido no Ollama e, se esse modelo não responder, reenvia o mesmo prompt para o Phi-3.5-mini.

O comando `python -m src` ainda abre o menu antigo de cadastro de modelos. A API desta branch sobe com `python -m src.loadbalancer`.

## Links

> [Artigo (Overleaf)](https://www.overleaf.com/5344664863vpbcpygzymjm#28a138)

> [Repositório (GitHub)](https://github.com/arturmartinho/Trabalho-Interdiciplinar-5)

> [Classroom 50](https://classroom50.org/ICEI-PUC-Minas-CC-TI/plmg-cc-2026-2-ti5/assignments/plmg-cc-2026-2-ti5/accept)

## Pré-requisitos

- Python 3.11 ou mais novo
- [Ollama](https://ollama.com/download/windows) no ar, em `http://127.0.0.1:11434`
- Os três modelos do trabalho:

```powershell
ollama pull phi3.5
ollama pull qwen2.5-coder:0.5b
ollama pull Alieno/ailo-152m-v2
```

O classificador fica em `src/classificador.py`, na branch `feat/classificador`. Sem esse arquivo, `POST /infer` responde 503. Nesta branch, traga o arquivo com:

```powershell
git checkout origin/feat/classificador -- src/classificador.py
```

## Como executar

Na pasta do projeto:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env
python -m src.loadbalancer
```

No Linux ou macOS, ative o ambiente com `source .venv/bin/activate` e copie o exemplo com `cp .env.example .env`.

A API escuta em `http://127.0.0.1:8000`. A documentação interativa fica em `http://127.0.0.1:8000/docs`.

O `.env` precisa repetir os tags de `ollama list`. Os valores esperados são:

```text
OLLAMA_HOST=http://127.0.0.1:11434
MODELO_SIMPLES=Alieno/ailo-152m-v2
MODELO_CODIGO=qwen2.5-coder:0.5b
MODELO_COMPLEXO=phi3.5
FALLBACK_MODEL=phi3.5
```

`OLLAMA_NUM_PREDICT` é opcional. Quando está definido, limita o tamanho da resposta. Sem essa variável, a geração segue até o fim, com timeout de `OLLAMA_TIMEOUT_SECONDS`.

## Endpoints

`GET /health` consulta o Ollama e devolve os modelos instalados e os três tags esperados.

`POST /infer` recebe um JSON com `prompt` e, se a medição precisar repetir o mesmo item, `prompt_id`. A resposta traz a classe, o motivo, o modelo escolhido, o modelo que atendeu, se houve redirecionamento e as latências em segundos (`attempt_latency_s`, `ollama_latency_s` e `e2e_latency_s`).

`POST /v1/chat/completions` aceita `messages` no formato de conversa, usa a última mensagem do usuário e ignora o campo `model`.

Um prompt vazio responde 422. As duas tentativas falhando respondem 502. Classificador ausente responde 503.

## Testes

Os testes de unidade e o teste com Ollama falso não precisam dos modelos instalados:

```powershell
python -m pytest tests/test_loadbalancer.py tests/test_loadbalancer_integracao.py -q
```

O teste contra o Ollama desta máquina só roda se os três modelos estiverem instalados. Ele limita cada resposta a 32 tokens:

```powershell
python -m pytest tests/test_loadbalancer_ollama_real.py -q
```

Para um caso manual, com a API no ar:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
$pedido = @{ prompt = "Qual é a capital da França?"; prompt_id = "s01" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/infer -ContentType "application/json; charset=utf-8" -Body $pedido
```

Pedido curto vai para o AILO, pedido de código para o Qwen e pedido longo ou de raciocínio para o Phi. O terminal da API mostra as linhas `despacho`, `resultado` e, quando houver desvio, `redirecionamento`.
