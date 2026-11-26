"""Normalização do uso de tokens e das durações reportadas pelo Ollama.

O Ollama devolve ``prompt_eval_count``/``eval_count`` e durações em
nanossegundos em ``POST /api/generate``. Este módulo é o único lugar que
conhece esses nomes: os contratos (`schemas.py`), o cliente HTTP e o runner
de experimentos reaproveitam a mesma conversão.
"""

from typing import Any, Mapping

TOKEN_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)
DURATION_FIELDS = (
    "prompt_eval_duration_s",
    "eval_duration_s",
    "load_duration_s",
    "total_duration_s",
)
USAGE_FIELDS = TOKEN_FIELDS + DURATION_FIELDS + ("token_source",)

_DURATION_KEYS = {
    "prompt_eval_duration_s": "prompt_eval_duration",
    "eval_duration_s": "eval_duration",
    "load_duration_s": "load_duration",
    "total_duration_s": "total_duration",
}


def _inteiro(valor: Any) -> int | None:
    if valor is None or isinstance(valor, bool):
        return None
    try:
        return int(valor)
    except (TypeError, ValueError):
        return None


def _segundos(valor: Any) -> float | None:
    ns = _inteiro(valor)
    if ns is None or ns < 0:
        return None
    return round(ns / 1_000_000_000, 6)


def extract_usage(payload: Any) -> dict[str, Any]:
    """Converte a resposta do Ollama em campos opcionais de uso.

    Campos ausentes viram ``None`` (ou não aparecem) para não inventar
    medição. ``total_tokens`` é calculado somente quando as duas contagens
    originais estão presentes.
    """
    if not isinstance(payload, Mapping):
        return {}

    usage: dict[str, Any] = {}
    prompt = _inteiro(payload.get("prompt_eval_count", payload.get("prompt_tokens")))
    completion = _inteiro(payload.get("eval_count", payload.get("completion_tokens")))
    if prompt is not None:
        usage["prompt_tokens"] = prompt
    if completion is not None:
        usage["completion_tokens"] = completion
    if prompt is not None and completion is not None:
        usage["total_tokens"] = prompt + completion
    else:
        total = _inteiro(payload.get("total_tokens"))
        if total is not None:
            usage["total_tokens"] = total

    for campo, chave in _DURATION_KEYS.items():
        valor = _segundos(payload.get(campo, payload.get(chave)))
        if valor is not None:
            usage[campo] = valor

    if "prompt_tokens" in usage or "completion_tokens" in usage:
        usage["token_source"] = "ollama"
    return usage


def usage_from_call(call: Any) -> dict[str, Any]:
    """Copia os campos de uso de ``OllamaCall`` (ou objeto compatível)."""
    usage = {campo: getattr(call, campo, None) for campo in USAGE_FIELDS}
    return {campo: valor for campo, valor in usage.items() if valor is not None}


def sem_usage(source: str) -> dict[str, Any]:
    """Contrato vazio com origem explícita (ex.: balancer sem os campos)."""
    return {campo: None for campo in TOKEN_FIELDS + DURATION_FIELDS} | {"token_source": source}
