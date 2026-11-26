#!/usr/bin/env bash
# Executa a rodada completa: checagens -> experimento -> análise -> relatório.
# Uso (com a venv ativa, balancer rodando em outro terminal):
#   ./executar_rodada.sh            # 3 repetições, 2 passadas de aquecimento
#   ./executar_rodada.sh 1 1        # rodada curta de teste
#   CONC=4 PROMPTS=prompts_experimento.json ./executar_rodada.sh 2 2
#   DEPLOYMENTS=local,remote SCENARIOS=default_model,loadbalancer_classifier \
#     OLLAMA_REMOTE_HOST=http://servidor:11434 LB_REMOTE_URL=http://servidor:8000 \
#     ./executar_rodada.sh 2 2
set -euo pipefail
cd "$(dirname "$0")"
REP="${1:-3}"
AQ="${2:-2}"
PY="${PYTHON:-python}"
PROMPTS="${PROMPTS:-prompts.json}"
CONC="${CONC:-1}"
SAIDA="${SAIDA:-resultados.csv}"
SCENARIOS="${SCENARIOS:-default_model,loadbalancer_classifier}"
DEPLOYMENTS="${DEPLOYMENTS:-local}"
# Padrão: maior modelo registrado no catálogo do LB (respeita MODELO_BASELINE).
MODELO_PADRAO="$(PYTHONPATH=.. "$PY" -c \
  'from experiments.config_experimento import modelo_padrao; print(modelo_padrao())' 2>/dev/null || true)"
MODEL_TAG="${MODEL_TAG:-${MODELO_PADRAO:-${FALLBACK_MODEL:-phi3.5}}}"
RESOURCE_SAMPLE_MS="${RESOURCE_SAMPLE_MS:-250}"
GPU_METRICS="${GPU_METRICS:-auto}"
RESUMO="${SAIDA%.csv}.resumo.csv"

"$PY" -c "import httpx, dotenv" 2>/dev/null || {
  echo "ERRO: ative a venv e instale as dependências (pip install -e '.[experiments]')."; exit 1; }

NP="${OLLAMA_NUM_PREDICT:-$(grep -E '^OLLAMA_NUM_PREDICT=' ../.env 2>/dev/null | tail -1 | cut -d= -f2 | tr -d '\r' || true)}"
if [ -z "$NP" ]; then
  echo "ERRO: OLLAMA_NUM_PREDICT não definido (nem no ambiente, nem em ../.env)."
  echo "Defina (ex.: echo 'OLLAMA_NUM_PREDICT=256' >> ../.env), reinicie o balancer e rode de novo."
  exit 1
fi

ARGS=(--repeticoes "$REP" --aquecimento "$AQ" --prompts "$PROMPTS"
      --concorrencia "$CONC" --saida "$SAIDA" --scenarios "$SCENARIOS"
      --deployments "$DEPLOYMENTS" --model-tag "$MODEL_TAG"
      --resource-sample-ms "$RESOURCE_SAMPLE_MS" --gpu-metrics "$GPU_METRICS")
[ -n "${OLLAMA_LOCAL:-}" ] && ARGS+=(--ollama-local "$OLLAMA_LOCAL")
[ -n "${OLLAMA_REMOTE_HOST:-}" ] && ARGS+=(--ollama-remote "$OLLAMA_REMOTE_HOST")
[ -n "${LB_LOCAL_URL:-}" ] && ARGS+=(--lb-local-url "$LB_LOCAL_URL")
[ -n "${LB_REMOTE_URL:-}" ] && ARGS+=(--lb-remote-url "$LB_REMOTE_URL")

echo "Limite de tokens: $NP (o balancer precisa ter sido iniciado DEPOIS de definir isso)"
echo "Matriz efetiva: cenários=$SCENARIOS deployments=$DEPLOYMENTS model_tag=$MODEL_TAG"
echo "Amostragem de recursos: ${RESOURCE_SAMPLE_MS}ms (gpu_metrics=$GPU_METRICS)"

if [ -s "$SAIDA" ]; then
  mkdir -p anteriores
  TS="$(date +%Y%m%d_%H%M%S)"
  BASE="$(basename "$SAIDA" .csv)"
  mv "$SAIDA" "anteriores/${BASE}_$TS.csv"
  [ -f "${SAIDA%.csv}.metadados.json" ] && mv "${SAIDA%.csv}.metadados.json" "anteriores/${BASE}_$TS.metadados.json"
  [ -f "$RESUMO" ] && mv "$RESUMO" "anteriores/${BASE}_$TS.resumo.csv"
  echo "CSV anterior guardado em anteriores/${BASE}_$TS.csv"
fi

"$PY" rodar_experimento.py "${ARGS[@]}" | tee log_rodada.txt
"$PY" analisar.py --entrada "$SAIDA" --resumo "$RESUMO" | tee -a log_rodada.txt
echo
echo "Pronto. Envie para a revisão: $SAIDA, ${SAIDA%.csv}.metadados.json, \
$RESUMO, relatorio_resultados.md e graficos/"
