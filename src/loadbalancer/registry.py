"""Ponte entre o catálogo de modelos (camada de dados) e o load balancer.

O ``AIModelDM`` guarda os modelos registrados pelo usuário, inclusive a
classe que cada um atende (``simples``, ``codigo``, ``complexo`` ou
``reserva``) e a URL do backend. Este módulo traduz esse catálogo para o
formato que o classificador e o despacho consomem, mantendo as variáveis
de ambiente como fallback quando o banco ainda não tem um modelo para a
classe.
"""

from __future__ import annotations

from typing import Any, Dict

from src.ai_model import AIModelDM, normalize_role
from src.loadbalancer.settings import (
    FALLBACK_MODEL,
    MODEL_CODE,
    MODEL_REASONING,
    MODEL_SIMPLE,
    OLLAMA_HOST,
)

CLASS_ROLES = ("simples", "codigo", "complexo")

DEFAULT_BY_ROLE = {
    "simples": MODEL_SIMPLE,
    "codigo": MODEL_CODE,
    "complexo": MODEL_REASONING,
}


def _same_model(left: str, right: str) -> bool:
    """Compara duas tags ignorando namespace e sufixo ``:tag``."""
    def base(value: str) -> str:
        return value.split(":", 1)[0].strip().lower().split("/")[-1]

    return base(left) == base(right)


def _rows() -> list[dict[str, Any]]:
    """Lê o catálogo sem derrubar o serviço se o banco estiver indisponível."""
    try:
        return AIModelDM.list()
    except Exception:
        return []


def _active(row: dict[str, Any]) -> bool:
    return str(row.get("status") or "").lower() == "up"


def _target(row: dict[str, Any]) -> dict[str, str]:
    return {
        "name": (row.get("name") or "").strip(),
        "url": (row.get("url") or OLLAMA_HOST).rstrip("/"),
    }


def model_targets() -> Dict[str, dict[str, str]]:
    """Devolve ``classe -> {name, url}`` para os modelos ativos registrados.

    Quando há mais de um modelo para a mesma classe, vence o primeiro
    registrado. Classes sem modelo no banco continuam usando o padrão do
    ambiente (ver :data:`DEFAULT_BY_ROLE`).
    """
    found: Dict[str, dict[str, str]] = {}
    for row in _rows():
        if not _active(row):
            continue
        role = normalize_role(row.get("role"))
        target = _target(row)
        if role in CLASS_ROLES and target["name"] and role not in found:
            found[role] = target
    return found


def model_roles() -> Dict[str, str]:
    """Devolve ``classe -> tag`` (atalho de :func:`model_targets`)."""
    return {role: target["name"] for role, target in model_targets().items()}


def model_urls() -> Dict[str, str]:
    """Devolve ``tag -> URL`` para todos os modelos ativos do catálogo."""
    return {
        _target(row)["name"]: _target(row)["url"]
        for row in _rows()
        if _active(row) and (row.get("name") or "").strip()
    }


def url_for_model(name: str) -> str:
    """URL do backend registrado para a tag, ou ``OLLAMA_HOST``."""
    for row in _rows():
        if _active(row) and _same_model(row.get("name") or "", name):
            return _target(row)["url"]
    return OLLAMA_HOST


def fallback_target() -> dict[str, str]:
    """Modelo de reserva: ``reserva`` registrado ou ``FALLBACK_MODEL``."""
    for row in _rows():
        if _active(row) and normalize_role(row.get("role")) == "reserva":
            target = _target(row)
            if target["name"]:
                return target

    for row in _rows():
        if _active(row) and _same_model(row.get("name") or "", FALLBACK_MODEL):
            return _target(row)

    return {"name": FALLBACK_MODEL, "url": OLLAMA_HOST}


def fallback_model() -> str:
    """Tag do modelo de reserva."""
    return fallback_target()["name"]


def expected_models() -> list[str]:
    """Lista de tags esperadas para as três classes, na ordem do contrato."""
    roles = model_roles()
    return [roles.get(role, DEFAULT_BY_ROLE[role]) for role in CLASS_ROLES]


def registered_models() -> list[dict[str, Any]]:
    """Catálogo completo, para exibição em ``/health`` e na CLI."""
    return [
        {
            "id": row.get("id"),
            "name": row.get("name"),
            "type": row.get("type"),
            "status": row.get("status"),
            "role": normalize_role(row.get("role")) or "geral",
            "url": row.get("url") or "",
        }
        for row in _rows()
    ]
