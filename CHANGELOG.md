# Changelog

Todas as mudanças relevantes do projeto são registradas aqui.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/)
e o projeto usa versionamento semântico.

## [0.2.0] — 2026-11-05

### Adicionado

- **CLI unificada** (`src/cli.py`, acessível por `python -m src` e pelo atalho
  `ai-lb`) com os comandos:
  - `models list/register/host/connect/update/stop/resume/call/remove`
  - `classify`, `infer`, `serve`, `experiment` e `analyze`
- **Ponte catálogo ↔ load balancer** (`src/loadbalancer/registry.py`): o
  load balancer agora consulta o `AIModelDM` para descobrir a tag e a URL
  de cada classe (`simples`, `codigo`, `complexo`, `reserva`).
- **Despacho por backend**: `OllamaClient.generate` aceita `base_url`, então
  modelos remotos cadastrados com `models connect` são realmente usados
  pelo load balancer, e não apenas exibidos no catálogo.
- **Comando `models register`** para cadastrar modelos locais/remotos sem
  exigir contato imediato com o backend.
- **Coluna `role`** na tabela `ai_models`, com migração automática de bancos
  SQLite já existentes.
- **Aliases de classe** (`fallback` → `reserva`, `code` → `codigo`, etc.).
- Testes para a CLI e para a leitura do catálogo pelo load balancer
  (`tests/test_cli.py`).
- **Artigo** (`article/main.tex`) atualizado com a arquitetura em camadas, o
  catálogo/CLI, as especificações da implementação e gráficos de latência
  derivados dos resultados preliminares. As figuras são geradas por
  `article/gerar_graficos.py`.

### Alterado

- `src/__main__.py` agora abre a CLI unificada (antes abria apenas o menu).
- `src/classificador.py` aceita um mapa opcional de modelos, permitindo que o
  load balancer use o catálogo do banco sem perder os padrões do `.env`.
- `src/loadbalancer/api.py` reporta no `/health` os modelos esperados,
  registrados e ausentes, e usa o modelo `reserva` do catálogo como fallback.
- `src/loadbalancer/dispatcher.py` deixou de manipular `sys.path` e passou a
  usar imports de pacote (`src.classificador`).
- Imports internos foram padronizados para o estilo de pacote
  (`from . import log`, `from src...`).
- `pyproject.toml` ganhou `build-system`, o console script `ai-lb`, o extra
  `experiments` (matplotlib, numpy, psutil) e configuração de descoberta de
  pacotes.
- README reescrito para refletir a CLI, a arquitetura em camadas e a nova
  localização dos experimentos.
- Scripts de experimento atualizados para orientar `python -m src serve`.

### Movido

- `src/resources/rodar_experimento.py` → `experiments/rodar_experimento.py`
- `src/resources/analisar.py` → `experiments/analisar.py`
- `src/resources/prompts.json` → `experiments/prompts.json`
- `src/resources/executar_rodada.sh` → `experiments/executar_rodada.sh`
- `src/resources/secao_resultados_preliminares.md` →
  `docs/secao_resultados_preliminares.md`

A pasta `experiments/` agora coincide com o que o README e o artigo descrevem,
e o carregamento do `.env` na raiz volta a funcionar.

### Removido

- `src/calculadora.py` e `tests/test_calculadora.py` — código de exemplo sem
  relação com o escopo do projeto.
- Exportações da calculadora em `src/__init__.py`.
- Pasta `src/resources/`, que misturava scripts operacionais com código-fonte.

### Corrigido

- `rodar_experimento.py` carregava o `.env` a partir de `src/` por causa da
  localização antiga; com a mudança para `experiments/`, o caminho voltou a
  apontar para a raiz do repositório.
- `executar_rodada.sh` procurava o `.env` em `../.env`; agora o caminho está
  correto a partir de `experiments/`.

### Compatibilidade

- `python -m src.loadbalancer` continua funcionando como alias do serviço.
- Os nomes `MODELO_SIMPLES`, `MODELO_CODIGO`, `MODELO_COMPLEXO` e
  `FALLBACK_MODEL` do `.env` continuam válidos como fallback.
