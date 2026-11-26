"""Schema CSV v2 dos experimentos.

As colunas antigas continuam nas mesmas posições; tudo que foi acrescentado
vem depois, para que planilhas e scripts antigos continuem lendo o arquivo.
"""

from __future__ import annotations

SCHEMA_VERSION = "2"

COLUNAS_V1 = [
    "timestamp", "rodada", "passo", "ordem_exec", "cenario", "prompt_id", "classe_esperada", "classe",
    "modelo_escolhido", "modelo_atendeu", "desvio", "latencia_s", "lb_e2e_s",
    "resposta_chars", "sucesso", "erro",
    "concorrencia", "idioma", "subtipo", "linguagem",
]

COLUNAS_V2 = COLUNAS_V1 + [
    # identidade/execução
    "schema_version", "run_id", "scenario", "deployment", "model_tag",
    "target_ollama_host", "target_lb_url",
    "effective_ollama_host", "effective_model_tag", "effective_lb_url",
    # classificação / loss do teste
    "classe_predita", "classification_correct", "classification_error",
    # latência
    "latency_e2e_s", "ollama_latency_s",
    # tokens e durações do Ollama
    "prompt_tokens", "completion_tokens", "total_tokens", "token_source",
    "prompt_eval_duration_s", "eval_duration_s", "load_duration_s", "total_duration_s",
    # RAM
    "ram_rss_mb_avg", "ram_rss_mb_peak",
    "ram_system_used_mb_avg", "ram_system_used_mb_peak",
    "ram_system_pct_avg", "ram_system_pct_peak",
    # GPU (vazio em CPU-only)
    "gpu_available", "gpu_count", "gpu_util_pct_avg", "gpu_util_pct_peak",
    "gpu_mem_used_mb_avg", "gpu_mem_used_mb_peak", "gpu_mem_total_mb",
    "gpu_power_w_avg", "gpu_power_w_peak", "gpu_sample_count", "gpu_sample_interval_ms",
    # degradação de amostragem
    "resource_error",
]

# Alias usado pelo runner e por quem importava a lista antiga.
COLUNAS = COLUNAS_V2


def linha_vazia() -> dict[str, str]:
    """Todas as colunas com valor vazio (o escritor preenche o que mediu)."""
    return {coluna: "" for coluna in COLUNAS_V2}
