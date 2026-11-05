#!/usr/bin/env bash
# Executa a rodada completa: checagens -> experimento -> análise -> relatório.
# Uso (com a venv ativa, balancer rodando em outro terminal):
#   ./executar_rodada.sh            # 3 repetições, 2 passadas de aquecimento
#   ./executar_rodada.sh 1 1        # rodada curta de teste
set -euo pipefail
cd "$(dirname "$0")"
REP="${1:-3}"
AQ="${2:-2}"
PY="${PYTHON:-python}"

"$PY" -c "import httpx, numpy, matplotlib, dotenv" 2>/dev/null || {
  echo "ERRO: ative a venv e instale: pip install matplotlib numpy"; exit 1; }

NP="${OLLAMA_NUM_PREDICT:-$(grep -E '^OLLAMA_NUM_PREDICT=' ../.env 2>/dev/null | tail -1 | cut -d= -f2 | tr -d '\r' || true)}"
if [ -z "$NP" ]; then
  echo "ERRO: OLLAMA_NUM_PREDICT não definido (nem no ambiente, nem em ../.env)."
  echo "Defina (ex.: echo 'OLLAMA_NUM_PREDICT=256' >> ../.env), reinicie o balancer e rode de novo."
  exit 1
fi
echo "Limite de tokens: $NP (o balancer precisa ter sido iniciado DEPOIS de definir isso)"

if [ -s resultados.csv ]; then
  mkdir -p anteriores
  TS="$(date +%Y%m%d_%H%M%S)"
  mv resultados.csv "anteriores/resultados_$TS.csv"
  [ -f resultados.metadados.json ] && mv resultados.metadados.json "anteriores/resultados_$TS.metadados.json"
  echo "CSV anterior guardado em anteriores/resultados_$TS.csv"
fi

"$PY" rodar_experimento.py --repeticoes "$REP" --aquecimento "$AQ" | tee log_rodada.txt
"$PY" analisar.py | tee -a log_rodada.txt
echo
echo "Pronto. Envie para a revisão: resultados.csv, resultados.metadados.json, relatorio_resultados.md e graficos/"
