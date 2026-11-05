# TI5 — Load Balancer para Modelos de IA

Roteamento de requisições por complexidade entre modelos de IA de diferentes portes, servidos via [Ollama](https://ollama.com/). O projeto reúne, em uma única CLI, o catálogo de modelos, o classificador de prompts, o serviço de balanceamento e os experimentos.

**Integrantes:**
- Artur Fernandes Braga de Menezes
- Artur Rizzi Martinho
- Felipe Lacerda Tertuliano
- Gabriel Alves da Silva Diógenes
- Rafael Mortimer Colares

**Instituição:** Pontifícia Universidade Católica de Minas Gerais (PUC Minas)
**Disciplina:** Trabalho Interdisciplinar V
**Sprint atual:** Sprint 2 (semana 3)

---

## 📖 Sobre o projeto

Sistemas de IA em produção costumam enviar **toda** requisição — simples ou complexa — para o **mesmo modelo de grande porte**. Isso gera uso ineficiente de GPU/CPU em tarefas triviais, maior latência e maior custo de inferência.

Este projeto propõe um **Load Balancer** que:

1. **Mantém um catálogo** de modelos locais e remotos, com a classe que cada um atende;
2. **Classifica** cada prompt pela complexidade (regras determinísticas);
3. **Encaminha** para o modelo mais adequado entre os registrados:
   - `AILO-152M-v2` → prompts **simples** (curtos, factuais)
   - `Qwen2.5-Coder-0.5B` → prompts de **código**
   - `Phi-3.5-mini` → prompts **complexos** (longos ou de raciocínio)
4. **Mede** latência e registra cada decisão;
5. **Executa e analisa** os experimentos baseline × proposta.

**Lacuna explorada:** os trabalhos relacionados (SAL, Llumnix, SkyWalker, DualMap) balanceiam **réplicas do mesmo modelo**. Nenhum classifica a complexidade do prompt para escolher entre **modelos de portes diferentes**.

---

## 🏗️ Arquitetura em camadas

O código segue uma separação simples em camadas, com uma única porta de entrada:

```
┌─────────────────────────────────────────────────────────────┐
│ Apresentação / CLI                                          │
│   python -m src ...   (ou ai-lb ...)                        │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│ Aplicação / serviço                                         │
│   Classificador · Despacho · API FastAPI · Cliente Ollama   │
│   src/classificador.py · src/loadbalancer/                  │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│ Domínio / catálogo                                          │
│   AIModelDM (modelo, tipo, status, URL e classe)            │
│   src/ai_model.py                                           │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│ Dados                                                       │
│   SQLite (ai_models) · src/data_access.py                   │
└─────────────────────────────────────────────────────────────┘
```

O **load balancer consulta o catálogo** (`AIModelDM`) para descobrir qual tag atende cada classe. Se uma classe ainda não tiver modelo ativo no banco, ele usa a tag correspondente do `.env` como fallback.

```
Cliente
   │
   │ POST /infer { prompt, prompt_id }
   ▼
┌──────────────────────────────────┐
│   Load Balancer (FastAPI)        │
│                                  │
│  1. Catálogo → classe → modelo   │
│  2. Classificador (regras)       │
│  3. Despacho via httpx           │
│  4. Fallback se o modelo falhar  │
└──────────────────────────────────┘
   │
   │ POST /api/generate
   ▼
┌──────────────────────────────────┐
│   Ollama (localhost:11434)       │
│  AILO / Qwen-Coder / Phi-3.5     │
└──────────────────────────────────┘
```

### Estrutura de pastas

```
.
├── article/                 # Artigo SBC (.tex, .pdf)
├── docs/                    # Cronograma e seções auxiliares
├── presentation/            # Slides
├── experiments/             # Scripts e dados do experimento
│   ├── rodar_experimento.py
│   ├── analisar.py
│   ├── prompts.json
│   └── executar_rodada.sh
├── src/
│   ├── __main__.py          # `python -m src`
│   ├── cli.py               # CLI unificada (models, serve, infer, ...)
│   ├── ai_model.py          # Catálogo de modelos (camada de domínio)
│   ├── data_access.py       # SQLite (camada de dados)
│   ├── classificador.py     # Classificador por regras
│   ├── log.py               # Logging colorido
│   └── loadbalancer/
│       ├── api.py           # Endpoints FastAPI
│       ├── dispatcher.py    # Classifica + despacha + fallback
│       ├── ollama_client.py # Cliente httpx do Ollama
│       ├── registry.py      # Ponte catálogo ↔ load balancer
│       ├── schemas.py       # Contratos (dataclasses)
│       └── settings.py      # Configuração via .env
├── tests/
│   ├── test_cli.py
│   ├── test_classificador.py
│   ├── test_loadbalancer.py
│   ├── test_loadbalancer_integracao.py
│   └── test_loadbalancer_ollama_real.py
├── .env.example
├── CHANGELOG.md
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## ⚙️ Pré-requisitos

- **Python** 3.11 ou superior
- **Ollama** instalado e rodando ([download](https://ollama.com/download))
- Os **três modelos** disponíveis no Ollama:
  ```bash
  ollama pull Alieno/ailo-152m-v2
  ollama pull qwen2.5-coder:0.5b
  ollama pull phi3.5
  ```
- **~4 GB de RAM livre** (para os três modelos pequenos)

> 💡 Confira os nomes das tags com `ollama list`. Eles precisam bater com o que for registrado na CLI.

---

## 🚀 Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/arturmartinho/Trabalho-Interdiciplinar-5.git
cd Trabalho-Interdiciplinar-5
```

### 2. Criar e ativar o ambiente virtual

```bash
python -m venv .venv
```

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

> Em alguns sistemas, use `python3` em vez de `python`.

### 3. Instalar as dependências

```bash
pip install -e ".[dev]"
```

Para rodar os experimentos (gráficos e métricas de RAM), instale também o extra:

```bash
pip install -e ".[dev,experiments]"
```

Alternativa sem instalação editável:

```bash
pip install -r requirements.txt
pip install -r requirements-experiments.txt
```

### 4. Configurar o `.env`

```bash
cp .env.example .env
```

Edite o `.env` conforme sua máquina. Os valores padrão já funcionam para uso local com Ollama.

### 5. Subir o Ollama

Em um terminal separado:

```bash
ollama serve
```

---

## 🖥️ CLI unificada

Tudo é acessível por `python -m src <comando>` (ou pelo atalho `ai-lb <comando>` depois de `pip install -e .`).

```bash
python -m src --help
python -m src models --help
```

| Comando | O que faz |
|---|---|
| `models list` | Lista os modelos registrados |
| `models register` | Registra um modelo manualmente (sem testar a conexão) |
| `models host` | Baixa/instala e sobe um modelo local no Ollama |
| `models connect` | Conecta a um modelo remoto e valida a URL |
| `models update` | Atualiza o status (ping) de um modelo ou de todos |
| `models stop` / `models resume` | Descarrega / recarrega um modelo local |
| `models call` | Envia um prompt direto a um modelo do catálogo |
| `models remove` | Remove um modelo do catálogo |
| `classify` | Mostra a classe e o modelo escolhido para um prompt |
| `infer` | Executa uma inferência pelo load balancer |
| `serve` | Sobe a API HTTP do load balancer |
| `experiment` | Roda baseline + proposta e grava `resultados.csv` |
| `analyze` | Gera métricas, gráficos e relatório a partir do CSV |

### Registrar os três modelos

A classe (`--role`) é o que liga cada modelo ao roteamento. Use `simples`, `codigo`, `complexo` e `reserva` (fallback).

```bash
# Modelos locais já baixados: registra o status como "up"
python -m src models register --name Alieno/ailo-152m-v2 --type local --role simples  --status up
python -m src models register --name qwen2.5-coder:0.5b  --type local --role codigo   --status up
python -m src models register --name phi3.5              --type local --role complexo --status up
python -m src models register --name phi3.5              --type local --role reserva  --status up
```

Se preferir que a CLI instale/baixe e suba o modelo no Ollama:

```bash
python -m src models host --name phi3.5 --role complexo
```

Para apontar para uma instância remota:

```bash
python -m src models connect --name phi3.5 --url http://192.168.0.10:11434 --role complexo
```

> **Precedência do roteamento:** modelo ativo no catálogo (`status=up`) para a classe → tag `MODELO_*` do `.env`. O fallback usa o modelo com `role=reserva`; se não houver, usa `FALLBACK_MODEL`.
>
> O catálogo também define o **endereço do backend**: um modelo registrado com `--url` é chamado nessa URL, e os modelos locais usam `OLLAMA_HOST`. Assim, `models connect` não serve apenas para listar — o load balancer despacha de fato para o nó remoto.
>
> O classificador resolve o catálogo na primeira requisição. Depois de registrar ou trocar um modelo, reinicie `python -m src serve`.

### Subir o load balancer

```bash
python -m src serve
# ou: python -m src serve --host 0.0.0.0 --port 8000
```

O serviço sobe em `http://127.0.0.1:8000`. Confira:

```bash
curl http://127.0.0.1:8000/health
```

> O alias legado `python -m src.loadbalancer` continua funcionando.

### Testar uma inferência pela CLI

Modo direto (não exige o `serve`; usa o mesmo despacho internamente):

```bash
python -m src infer --prompt "Qual é a capital da França?" --prompt-id teste-1
```

Contra um load balancer já em execução:

```bash
python -m src infer --url http://127.0.0.1:8000 --prompt "Escreva uma função em Python"
```

Use `--json` para ver classe, modelo escolhido, modelo que atendeu, redirecionamento e latências.

### Classificar sem executar o modelo

```bash
python -m src classify --prompt "Explique passo a passo por que o céu é azul." --json
```

### Rodar os testes

```bash
pytest
```

Para rodar apenas os testes que **não precisam** do Ollama:

```bash
pytest tests/test_classificador.py tests/test_loadbalancer.py tests/test_cli.py
```

Os testes de integração (`test_loadbalancer_integracao.py` e `test_loadbalancer_ollama_real.py`) sobem a API de verdade. O segundo é pulado automaticamente se o Ollama local não tiver os três modelos.

---

## 📡 API do Load Balancer

### `GET /health`

Verifica se o serviço está pronto (Ollama acessível + modelos esperados presentes).

**Resposta 200** (ok):
```json
{
  "ok": true,
  "ollama_host": "http://127.0.0.1:11434",
  "fallback_model": "phi3.5",
  "expected_models": ["Alieno/ailo-152m-v2", "qwen2.5-coder:0.5b", "phi3.5"],
  "models": ["Alieno/ailo-152m-v2", "qwen2.5-coder:0.5b", "phi3.5"],
  "registered_models": [
    {"id": 1, "name": "phi3.5", "type": "local", "status": "up", "role": "complexo", "url": ""}
  ],
  "missing_models": [],
  "error": null
}
```

**Resposta 503** (Ollama offline ou modelo esperado ausente):
```json
{ "ok": false, "error": "..." }
```

### `POST /infer`

Recebe um prompt, classifica e encaminha.

**Request:**
```json
{
  "prompt": "Qual é a capital da França?",
  "prompt_id": "s01"
}
```

**Resposta 200:**
```json
{
  "request_id": "a1b2c3d4e5f6",
  "prompt_id": "s01",
  "response": "Paris",
  "label": "simples",
  "reason": "pedido curto (5 palavras <= 25) sem sinais de codigo/raciocinio",
  "chosen_model": "Alieno/ailo-152m-v2",
  "served_model": "Alieno/ailo-152m-v2",
  "redirected": false,
  "redirect_reason": null,
  "attempt_latency_s": 0.312,
  "ollama_latency_s": 0.312,
  "e2e_latency_s": 0.320,
  "ok": true,
  "error": null
}
```

### `POST /v1/chat/completions`

Compatível com o formato OpenAI (simplificado). O campo `model` é **ignorado** — quem decide é o classificador.

```json
{
  "messages": [
    {"role": "user", "content": "Escreva uma função em Python"}
  ]
}
```

---

## 🧠 Classificador de prompts

Arquivo: `src/classificador.py`. Função pura `classificar(prompt, modelos=None)` — mesma entrada, mesma saída.

**Regras (a primeira que casar decide):**

| # | Condição | Classe | Modelo padrão |
|---|----------|--------|---------------|
| 1 | Prompt vazio | `simples` | AILO |
| 2 | Mais de **120 palavras** | `complexo` | Phi |
| 3 | Sinal de **código** | `codigo` | Qwen |
| 4 | Sinal de **raciocínio** | `complexo` | Phi |
| 5 | Até **25 palavras**, sem sinais | `simples` | AILO |
| 6 | Demais casos (ambíguo) | `complexo` | Phi |

**Casos ambíguos tratados:**
- Código + raciocínio → **código tem prioridade** (o modelo especializado conhece melhor o domínio)
- Tamanho médio sem sinais → **Phi** (errar pro maior aumenta latência, mas não compromete a resposta)

O parâmetro opcional `modelos` permite substituir os destinos do `.env` pelos modelos registrados no catálogo. É assim que o load balancer usa o `ai_model`.

---

## 🧪 Experimentos

O experimento compara **dois cenários** sobre o mesmo conjunto de prompts:

| Cenário | Descrição |
|---------|-----------|
| **Baseline** | Toda requisição vai para o `Phi-3.5-mini` |
| **Proposta** | O classificador decide o modelo |

### Passo a passo

1. Confirme que o Ollama e o load balancer estão no ar.
2. Rode a coleta:

   ```bash
   python -m src experiment --repeticoes 3 --aquecimento 2
   ```

3. Gere a análise:

   ```bash
   python -m src analyze
   ```

   Saídas em `experiments/`:
   - `resultados.csv` — uma linha por requisição
   - `resultados.metadados.json` — ambiente, versões, config
   - `relatorio_resultados.md` — relatório descritivo
   - `graficos/` — gráficos comparativos

Também é possível executar os scripts diretamente:

```bash
cd experiments
python rodar_experimento.py --repeticoes 3 --aquecimento 2
python analisar.py
```

### Argumentos do experimento

| Argumento | Padrão | Descrição |
|-----------|--------|-----------|
| `--repeticoes` | 1 | Número de rodadas completas por cenário |
| `--aquecimento` | 2 | Passadas descartadas (carrega os modelos em RAM) |
| `--seed` | 42 | Semente da ordem dos prompts (reprodutibilidade) |
| `--anexar` | false | Permite acrescentar a um CSV existente |
| `--prompts` | `prompts.json` | Caminho do conjunto de prompts |
| `--saida` | `resultados.csv` | Caminho do CSV de saída |

### Delineamento

Para evitar viés de ordem, o script:
- Executa o mesmo prompt nos **dois cenários**, uma vez cada por rodada;
- **Contrabalança** quem vai primeiro (metade baseline, metade proposta, dentro de cada classe);
- Separa os dois pedidos do mesmo prompt por ≥ n/2 chamadas;
- Faz **aquecimento** antes de medir, descartando essas chamadas.

### Formato do CSV

| Coluna | Descrição |
|--------|-----------|
| `timestamp` | Momento UTC da medição |
| `rodada` | Número da rodada |
| `passo` | 1 = primeiro cenário da rodada; 2 = o outro |
| `ordem_exec` | Ordem de execução dentro da rodada |
| `cenario` | `baseline` ou `proposta` |
| `prompt_id` | ID do prompt (ex: `S01`, `C03`, `X07`) |
| `classe_esperada` | `simples`, `codigo` ou `complexo` |
| `classe` | Classe **decidida** pelo classificador (vazio no baseline) |
| `modelo_escolhido` | Modelo que o classificador escolheu |
| `modelo_atendeu` | Modelo que de fato respondeu (pode diferir se houve fallback) |
| `desvio` | `true` se houve fallback |
| `latencia_s` | Latência de ponta a ponta medida no cliente |
| `lb_e2e_s` | Latência total reportada pelo Load Balancer |
| `resposta_chars` | Tamanho da resposta em caracteres |
| `sucesso` | `true` / `false` |
| `erro` | Mensagem de erro, se houver |

---

## 🗓️ Cronograma

O cronograma completo está em [`docs/schedule.md`](docs/schedule.md).

| Semana | Atividade |
|--------|-----------|
| 1 | Configuração do ambiente (Ollama, modelos leves) |
| 2 | Classificador de prompts + versão inicial do Load Balancer |
| **3 (Sprint 2)** | **Detalhamento da implementação + resultados preliminares + apresentação** |
| 4 | Refinamento do classificador + métricas + metodologia |
| 5 | Experimentos comparativos + análise estatística |
| 6 | Otimização + testes de roteamento + seção de resultados |
| 7 | Comparação com trabalhos relacionados |
| 8 | Conclusões + revisão do artigo |
| 9 (Sprint 3) | Pitch comercial + apresentação final |
| 10 | Submissão final + documentação |

---

## 🔗 Links

- **Artigo (Overleaf):** *(adicionar link)*
- **Apresentação (Slides):** `presentation/main.pptx`
- **GitHub Classroom:** *(adicionar link)*
- **Changelog:** [`CHANGELOG.md`](CHANGELOG.md)

---

## 📚 Referências principais

- Kossmann et al. (2024) — *Server-Aware Load Balancer (SAL)*
- Jain et al. (2025) — *Load balancing entre réplicas do mesmo modelo*
- Sun et al. (2024) — *Llumnix: reagendamento dinâmico de requisições*
- Xia et al. (2025) — *SkyWalker: balanceamento entre regiões geográficas*
- Yuan et al. (2026) — *DualMap: cache affinity + balanceamento*

---

## 📝 Licença

Projeto acadêmico — PUC Minas, Trabalho Interdisciplinar V.
