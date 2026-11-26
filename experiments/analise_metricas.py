"""Agregações puras dos resultados de experimentos (sem numpy/matplotlib).

Os nomes de cenário ``baseline``/``proposta`` de CSVs antigos são
normalizados para ``default_model``/``loadbalancer_classifier``; colunas
ausentes ficam vazias em vez quebrar a leitura.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

CENARIO_PADRAO = "default_model"
CENARIO_LB = "loadbalancer_classifier"
CENARIOS = (CENARIO_PADRAO, CENARIO_LB)
CLASSES = ("simples", "codigo", "complexo")
LEGENDS = {"baseline": CENARIO_PADRAO, "proposta": CENARIO_LB}

COLUNAS_RESUMO = [
    "scenario", "deployment", "classe_esperada", "n", "ok", "fail",
    "latency_mean_s", "latency_median_s", "latency_p95_s", "latency_p99_s", "latency_max_s",
    "ram_rss_mb_avg", "ram_rss_mb_peak", "ram_system_used_mb_avg", "ram_system_used_mb_peak",
    "ram_system_pct_avg", "ram_system_pct_peak",
    "gpu_available", "gpu_count", "gpu_util_pct_avg", "gpu_util_pct_peak",
    "gpu_mem_used_mb_avg", "gpu_mem_used_mb_peak", "gpu_mem_total_mb",
    "gpu_power_w_avg", "gpu_power_w_peak", "gpu_sample_count", "gpu_sample_interval_ms",
    "prompt_tokens_avg", "completion_tokens_avg", "total_tokens_sum", "token_rows",
    "classification_error_rate", "n_classified",
]

_RAM = ("ram_rss_mb", "ram_system_used_mb", "ram_system_pct")
_GPU = ("gpu_util_pct", "gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_power_w")
_GPU_PEAK = ("gpu_util_pct", "gpu_mem_used_mb", "gpu_power_w")


def valor(linha: Mapping[str, Any], *chaves: str) -> str:
    """Primeiro campo não vazio entre os nomes possíveis (compatibilidade)."""
    for chave in chaves:
        bruto = linha.get(chave)
        if bruto is not None and str(bruto).strip():
            return str(bruto).strip()
    return ""


def normalizar_cenario(bruto: str) -> str:
    return LEGENDS.get(bruto.lower(), bruto.lower())


def normalizar_deployment(bruto: str) -> str:
    return bruto.lower() if bruto.lower() in ("local", "remote") else "local"


def sucesso(linha: Mapping[str, Any]) -> bool:
    return valor(linha, "sucesso").lower() in ("true", "1", "sim")


def latencia(linha: Mapping[str, Any]) -> float | None:
    """Latência ponta a ponta: prefere ``latency_e2e_s`` (legado: ``latencia_s``)."""
    bruto = valor(linha, "latency_e2e_s", "latencia_s")
    if not bruto:
        return None
    try:
        return float(bruto)
    except ValueError:
        return None


def numero(linha: Mapping[str, Any], *chaves: str) -> float | None:
    bruto = valor(linha, *chaves)
    if not bruto:
        return None
    try:
        return float(bruto)
    except ValueError:
        return None


def classification_error(linha: Mapping[str, Any]) -> int | None:
    """Erro de classificação (loss do teste), ou None quando não se aplica.

    ``default_model`` não tem classificador: ``classe_predita`` fica vazia e a
    linha não entra no denominador. Em CSV v2 a coluna já vem calculada; em
    CSV antigo ela é derivada de ``classe``.
    """
    cenario = normalizar_cenario(valor(linha, "scenario", "cenario"))
    if cenario != CENARIO_LB:
        return None
    explicito = valor(linha, "classification_error")
    if explicito.lower() in ("0", "1"):
        return int(explicito)
    prevista = valor(linha, "classe_predita", "classe")
    esperada = valor(linha, "classe_esperada")
    if not prevista or not esperada:
        return None
    return 0 if prevista == esperada else 1


def percentil(valores: list[float], p: float) -> float:
    """Percentil com interpolação linear (mesma convenção de numpy)."""
    if not valores:
        return 0.0
    ordenados = sorted(valores)
    if len(ordenados) == 1:
        return ordenados[0]
    posicao = (len(ordenados) - 1) * p / 100
    baixo = int(posicao)
    alto = min(baixo + 1, len(ordenados) - 1)
    frac = posicao - baixo
    return ordenados[baixo] + frac * (ordenados[alto] - ordenados[baixo])


def metricas(valores: Iterable[float]) -> dict[str, float]:
    v = [float(x) for x in valores]
    if not v:
        return {k: 0.0 for k in ("media", "mediana", "dp", "p95", "p99", "max")}
    media = sum(v) / len(v)
    return {
        "media": media,
        "mediana": percentil(v, 50),
        "dp": (sum((x - media) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0,
        "p95": percentil(v, 95),
        "p99": percentil(v, 99),
        "max": max(v),
    }


def _novo_acumulador() -> dict[str, Any]:
    ac: dict[str, Any] = {
        "n": 0, "ok": 0, "fail": 0, "lat": [], "tokens": defaultdict(list),
        "erros": [], "resource_error": [], "gpu_available": False,
        "gpu_count": 0, "gpu_sample_count": 0, "gpu_sample_interval_ms": 0,
    }
    for chave in _RAM + _GPU:
        ac[f"{chave}__media"] = []
        ac[f"{chave}__pico"] = []
    return ac


def _acumular(ac: dict[str, Any], linha: Mapping[str, Any]) -> None:
    ac["n"] += 1
    if sucesso(linha):
        ac["ok"] += 1
        x = latencia(linha)
        if x is not None:
            ac["lat"].append(x)
    else:
        ac["fail"] += 1

    for chave, destino in (("prompt_tokens", "prompt_tokens"),
                           ("completion_tokens", "completion_tokens"),
                           ("total_tokens", "total_tokens")):
        v = numero(linha, chave)
        if v is not None:
            ac["tokens"][destino].append(v)

    erro = classification_error(linha)
    if erro is not None:
        ac["erros"].append(erro)

    motivo = valor(linha, "resource_error")
    if motivo:
        ac["resource_error"].append(motivo)
    if valor(linha, "gpu_available").lower() == "true":
        ac["gpu_available"] = True
    ac["gpu_count"] = max(ac["gpu_count"], int(numero(linha, "gpu_count") or 0))
    ac["gpu_sample_count"] += int(numero(linha, "gpu_sample_count") or 0)
    ac["gpu_sample_interval_ms"] = int(numero(linha, "gpu_sample_interval_ms") or 0)

    for chave in _RAM + _GPU:
        media = numero(linha, f"{chave}_avg")
        pico = numero(linha, f"{chave}_peak")
        if media is not None:
            ac[f"{chave}__media"].append(media)
        if pico is not None:
            ac[f"{chave}__pico"].append(pico)


def _media(lista: list[float]) -> float | None:
    return round(sum(lista) / len(lista), 3) if lista else None


def _pico(lista: list[float]) -> float | None:
    return round(max(lista), 3) if lista else None


def _finalizar(
    cenario: str, deployment: str, classe: str, ac: dict[str, Any]
) -> dict[str, Any]:
    lat = metricas(ac["lat"])
    res: dict[str, Any] = {
        "scenario": cenario, "deployment": deployment, "classe_esperada": classe,
        "n": ac["n"], "ok": ac["ok"], "fail": ac["fail"],
        "latency_mean_s": round(lat["media"], 3),
        "latency_median_s": round(lat["mediana"], 3),
        "latency_p95_s": round(lat["p95"], 3),
        "latency_p99_s": round(lat["p99"], 3),
        "latency_max_s": round(lat["max"], 3),
        "prompt_tokens_avg": _media(ac["tokens"]["prompt_tokens"]),
        "completion_tokens_avg": _media(ac["tokens"]["completion_tokens"]),
        "total_tokens_sum": round(sum(ac["tokens"]["total_tokens"]), 3) if ac["tokens"]["total_tokens"] else None,
        "token_rows": len(ac["tokens"]["total_tokens"]),
        "classification_error_rate": round(sum(ac["erros"]) / len(ac["erros"]), 4) if ac["erros"] else None,
        "n_classified": len(ac["erros"]),
        "gpu_available": ac["gpu_available"],
        "gpu_count": ac["gpu_count"],
        "gpu_sample_count": ac["gpu_sample_count"],
        "gpu_sample_interval_ms": ac["gpu_sample_interval_ms"],
    }
    for chave in _RAM + _GPU_PEAK:
        res[f"{chave}_avg"] = _media(ac[f"{chave}__media"])
        res[f"{chave}_peak"] = _pico(ac[f"{chave}__pico"])
    res["gpu_mem_total_mb"] = _pico(ac["gpu_mem_total_mb__media"])
    res["resource_error"] = "; ".join(sorted(set(ac["resource_error"])))
    return res


def agregar(linhas: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Agrega por cenário×deployment×classe, com subtotais e total global."""
    acumuladores: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(_novo_acumulador)
    for linha in linhas:
        cenario = normalizar_cenario(valor(linha, "scenario", "cenario"))
        deployment = normalizar_deployment(valor(linha, "deployment", "local"))
        classe = valor(linha, "classe_esperada") or "desconhecida"
        chaves = (
            (cenario, deployment, classe),
            (cenario, deployment, "todos"),
            (cenario, "todos", "todos"),
            ("todos", "todos", "todos"),
        )
        for chave in chaves:
            _acumular(acumuladores[chave], linha)
    return {
        chave: _finalizar(*chave, ac) for chave, ac in acumuladores.items()
    }


def avisos(
    linhas: list[Mapping[str, Any]],
    meta: Mapping[str, Any],
    grupos: Mapping[tuple[str, str, str], dict[str, Any]],
) -> list[str]:
    """Avisos operacionais e de escopo das métricas."""
    out: list[str] = []
    cfg = meta.get("config", {}) or {}
    if meta and not cfg.get("num_predict"):
        out.append("`OLLAMA_NUM_PREDICT` não estava definido: a latência mistura velocidade do modelo com o tamanho da resposta gerada.")
    if not meta:
        out.append("Sem `metadados.json`: não dá para confirmar limite de tokens, aquecimento, ambiente nem modelos carregados.")
    falhas = sum(int(r.get("fail") or 0) for k, r in grupos.items() if k == ("todos", "todos", "todos"))
    if falhas:
        out.append(f"Houve {falhas} falha(s); elas ficaram fora das latências.")
    total = grupos.get(("todos", "todos", "todos"), {})
    if not total.get("token_rows"):
        out.append("Token usage indisponível: o Ollama/LB não devolveu `prompt_eval_count`/`eval_count` ou a API é mais antiga.")
    if not total.get("gpu_available"):
        out.append("GPU indisponível (CPU-only ou `nvidia-smi` ausente): campos de GPU ficaram vazios.")
    if any(r.get("resource_error") for r in grupos.values()):
        out.append("A amostragem de recursos teve leituras degradadas; veja `resource_error` no CSV/metadados.")
    if any(
        k[0] == CENARIO_PADRAO and r.get("n") for k, r in grupos.items()
    ):
        out.append("`loss` = taxa de erro de classificação: `default_model` não tem classificador e não entra no denominador.")
    if total.get("n", 0) and total.get("n", 0) < 100:
        out.append(f"Com n={total['n']} requisições, P95/P99 são descritivos e não estimativas estáveis.")
    return out
