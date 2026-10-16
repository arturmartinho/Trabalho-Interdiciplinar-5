# TI5 — Load Balancer para Modelos de IA

Roteamento de requisições por complexidade entre modelos de IA de diferentes portes, servidos via [Ollama](https://ollama.com/).

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

1. **Classifica** cada prompt pela complexidade (regras determinísticas);
2. **Encaminha** para o modelo mais adequado entre três:
   - `AILO-152M-v2` → prompts **simples** (curtos, factuais)
   - `Qwen2.5-Coder-0.5B` → prompts de **código**
   - `Phi-3.5-mini` → prompts **complexos** (longos ou de raciocínio)
3. **Mede** latência, uso de RAM/GPU e registra cada decisão.

**Lacuna explorada:** os trabalhos relacionados (SAL, Llumnix, SkyWalker, DualMap) balanceiam **réplicas do mesmo modelo**. Nenhum classifica a complexidade do prompt para escolher entre **modelos de portes diferentes**.

---

## 🏗️ Arquitetura

```
Cliente
   │
   │ POST /infer { prompt, prompt_id }
   ▼
┌──────────────────────────────────┐
│   Load Balancer (FastAPI)        │
│                                  │
│  1. Classificador (regras)       │
│     └─ decide o modelo           │
│  2. Despacho                     │
│     └─ chama o Ollama via httpx  │
│  3. Fallback                     │
│     └─ se falhar, vai pro Phi    │
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
├── article/                # Artigo SBC (.tex, .pdf)
├── docs/                   # Documentação auxiliar
├── presentation/           # Slides da Sprint 2
├── experiments/            # Scripts e dados do experimento
│   ├── rodar_experimento.py
│   ├── prompts.json
│   └── resultados.csv      # (gerado ao rodar)
├── src/
│   ├── loadbalancer/       # Núcleo do Load Balancer
│   │   ├── api.py          # Endpoints FastAPI
│   │   ├── dispatcher.py   # Classifica + despacha + fallback
│   │   ├── ollama_client.py# Cliente httpx do Ollama
│   │   ├── schemas.py      # Contratos (dataclasses)
│   │   ├── settings.py     # Configuração via .env
│   │   ├── __init__.py
│   │   └── __main__.py     # `python -m src.loadbalancer`
│   ├── classificador.py    # Classificador por regras
│   ├── log.py              # Logging colorido
│   ├── ai_model.py         # Modelos (legado / menu)
│   ├── data_access.py      # Acesso SQLite (legado / menu)
│   ├── menu.py             # Menu CLI (legado)
│   └── calculadora.py      # Exemplo / teste inicial
├── tests/                  # Testes automatizados
├── .env.example            # Modelo do .env
├── .gitignore
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

> 💡 Confira os nomes das tags com `ollama list`. Eles precisam bater com as variáveis `MODELO_*` no `.env`.

---

## 🚀 Como executar

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

> Isso instala o pacote em modo editável + as dependências de desenvolvimento (pytest).
> Alternativa: `pip install -r requirements.txt`

### 4. Configurar o `.env`

```bash
cp .env.example .env
```

Edite o `.env` conforme sua máquina. Os valores padrão já funcionam para uso local com Ollama.

### 5. Subir o Ollama com os três modelos

Em um terminal separado:

```bash
ollama serve
```

E, em outro:

```bash
ollama pull Alieno/ailo-152m-v2
ollama pull qwen2.5-coder:0.5b
ollama pull phi3.5
```

> Confirme com `ollama list` que os três aparecem.

### 6. Subir o Load Balancer

```bash
python -m src.loadbalancer
```

O serviço sobe em `http://127.0.0.1:8000`. Confira em:

```bash
curl http://127.0.0.1:8000/health
```

Resposta esperada:
```json
{
  "ok": true,
  "ollama_host": "http://127.0.0.1:11434",
  "fallback_model": "phi3.5",
  "expected_models": ["Alieno/ailo-152m-v2", "qwen2.5-coder:0.5b", "phi3.5"],
  "models": ["Alieno/ailo-152m-v2", "qwen2.5-coder:0.5b", "phi3.5"],
  "error": null
}
```

### 7. Testar manualmente

**Prompt simples** (deve ir pro AILO):
```bash
curl -X POST http://127.0.0.1:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Qual é a capital da França?", "prompt_id": "teste-1"}'
```

**Prompt de código** (deve ir pro Qwen):
```bash
curl -X POST http://127.0.0.1:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Escreva uma função em Python que inverte uma string.", "prompt_id": "teste-2"}'
```

**Prompt complexo** (deve ir pro Phi):
```bash
curl -X POST http://127.0.0.1:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explique passo a passo por que o céu é azul.", "prompt_id": "teste-3"}'
```

### 8. Rodar os testes

```bash
pytest
```

Para rodar apenas os testes que **não precisam** do Ollama:

```bash
pytest tests/test_classificador.py tests/test_loadbalancer.py
```

---

## 🧪 Como reproduzir o experimento

O experimento compara **dois cenários** sobre o mesmo conjunto de prompts:

| Cenário | Descrição |
|---------|-----------|
| **Baseline** | Toda requisição vai para o `Phi-3.5-mini` |
| **Proposta** | O classificador decide o modelo |

### Passo a passo

1. **Confirme que o Ollama e o Load Balancer estão no ar** (passos 5 e 6 acima).
2. **Rode o script**:
   ```bash
   cd experiments
   python rodar_experimento.py --repeticoes 3 --aquecimento 2
   ```
3. **Saída esperada**:
   - `experiments/resultados.csv` — uma linha por requisição
   - `experiments/resultados.metadados.json` — ambiente, versões, config

### Argumentos do script

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

O arquivo `resultados.csv` tem as seguintes colunas:

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

### Análise dos resultados

Para gerar estatísticas (média, P95, P99) e gráficos, use um notebook ou script separado. Exemplo mínimo:

```python
import pandas as pd

df = pd.read_csv("experiments/resultados.csv")
df = df[df["sucesso"] == True]  # só requisições bem-sucedidas

for cenario in ("baseline", "proposta"):
    sub = df[df["cenario"] == cenario]["latencia_s"]
    print(f"{cenario}: média={sub.mean():.3f}s  P95={sub.quantile(0.95):.3f}s  P99={sub.quantile(0.99):.3f}s")
```

---

## 📡 API do Load Balancer

### `GET /health`

Verifica se o serviço está pronto (Ollama acessível + modelos disponíveis).

**Resposta 200** (ok):
```json
{
  "ok": true,
  "ollama_host": "http://127.0.0.1:11434",
  "fallback_model": "phi3.5",
  "expected_models": ["Alieno/ailo-152m-v2", "qwen2.5-coder:0.5b", "phi3.5"],
  "models": ["Alieno/ailo-152m-v2", "qwen2.5-coder:0.5b", "phi3.5"],
  "error": null
}
```

**Resposta 503** (Ollama offline ou modelo faltando):
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

**Resposta 502** (modelo escolhido e fallback falharam):
```json
{ "ok": false, "redirected": true, "served_model": "phi3.5", "response": null, "error": "..." }
```

### `POST /v1/chat/completions`

Compatível com o formato OpenAI (simplificado). O campo `model` é **ignorado** — quem decide é o classificador.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Escreva uma função em Python"}
  ]
}
```

---

## 🧠 Classificador de prompts

Arquivo: `src/classificador.py`. Função pura `classificar(prompt)` — mesma entrada, mesma saída.

**Regras (a primeira que casar decide):**

| # | Condição | Classe | Modelo |
|---|----------|--------|--------|
| 1 | Prompt vazio | `simples` | AILO |
| 2 | Mais de **120 palavras** | `complexo` | Phi |
| 3 | Sinal de **código** | `codigo` | Qwen |
| 4 | Nível de **Bloom ≥ 2** | `complexo` | Phi |
| 5 | Até **25 palavras**, nível Lembrar | `simples` | AILO |
| 6 | Demais casos (ambíguo) | `complexo` | Phi |

**Nível de Bloom:** o verbo de comando do prompt indica o nível cognitivo pela Taxonomia de Bloom revisada (1 Lembrar, 2 Entender, 3 Aplicar, 4 Analisar, 5 Avaliar, 6 Criar). Vale o maior nível encontrado; sem verbo, o nível é 1. Verbos no pretérito são ignorados ("Quem escreveu…?" é Lembrar). O limiar é configurável por `LIMIAR_BLOOM` no `.env` (padrão `2`).

**Casos ambíguos tratados:**
- Código + Bloom alto → **código tem prioridade** (o modelo especializado conhece melhor o domínio)
- Tamanho médio sem sinais → **Phi** (errar pro maior aumenta latência, mas não compromete a resposta)

**Avaliar a acurácia** sobre um conjunto rotulado (CSV ou JSON com `id`, `prompt` e `classe_esperada`):

```bash
python src/avaliar_classificador.py                              # data/prompts_rotulados.csv (73 prompts)
python src/avaliar_classificador.py src/resources/prompts.json   # conjunto do experimento (40 prompts)
```

---

## 🧪 Testes

```bash
pytest                                    # tudo
pytest tests/test_classificador.py        # só o classificador
pytest tests/test_loadbalancer.py         # só o LB (sem Ollama real)
pytest -v                                 # verboso
```

Os testes de integração (`test_api_integracao_real.py`) **só rodam se o Ollama estiver no ar** com os três modelos. Caso contrário, são pulados automaticamente.

---

## 🗓️ Cronograma

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
