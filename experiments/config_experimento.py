"""Configuração compartilhada dos scripts de experimento.

Carrega o `.env` da raiz uma única vez e centraliza os valores padrão usados
tanto pelo runner quanto pelos contratos do load balancer.

O modelo padrão dos experimentos é o maior modelo **registrado no catálogo do
load balancer** (``ai_models``). O tamanho vem dos metadados do Ollama
(``details.parameter_size``, com ``size`` em bytes como desempate). Um valor
explícito em ``MODELO_BASELINE`` continua tendo precedência.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

# Os scripts também rodam de dentro de experiments/; o catálogo do balancer
# precisa ser lido do DB da raiz, não de um arquivo novo criado no cwd.
_db_arquivo = os.getenv("DB_FILE", "").strip()
if _db_arquivo and not Path(_db_arquivo).is_absolute():
    os.environ["DB_FILE"] = str(REPO_ROOT / _db_arquivo)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_REMOTE_HOST = os.getenv("OLLAMA_REMOTE_HOST", "").strip().rstrip("/")
LB_URL = os.getenv("LB_URL", "http://127.0.0.1:8000").rstrip("/")
LB_LOCAL_URL = os.getenv("LB_LOCAL_URL", "").strip().rstrip("/") or LB_URL
LB_REMOTE_URL = os.getenv("LB_REMOTE_URL", "").strip().rstrip("/")

MODELO_FALLBACK = os.getenv("FALLBACK_MODEL", "").strip() or "phi3.5"
# Compatibilidade com quem importa a constante: valor explícito ou fallback.
MODELO_BASELINE = os.getenv("MODELO_BASELINE", "").strip() or MODELO_FALLBACK
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
# Na proposta o servidor pode tentar o modelo escolhido e depois o fallback (2x timeout).
TIMEOUT = OLLAMA_TIMEOUT * 2 + 60
NUM_PREDICT = os.getenv("OLLAMA_NUM_PREDICT", "").strip()
RESOURCE_SAMPLE_MS = int(os.getenv("RESOURCE_SAMPLE_MS", "250"))
GPU_METRICS = os.getenv("GPU_METRICS", "auto").strip().lower()


def _catalogo() -> list[dict[str, Any]]:
    """Linhas do catálogo do balancer (injeção de dependência nos testes)."""
    from src.loadbalancer.registry import registered_models

    return registered_models()


def _mesmo_modelo(esquerda: str, direita: str) -> bool:
    """Compara tags ignorando namespace e sufixo ``:tag``."""
    def base(valor: str) -> str:
        return valor.split(":", 1)[0].strip().lower().split("/")[-1]

    return base(esquerda) == base(direita)


def _parametros(item: dict[str, Any]) -> float:
    """Número de parâmetros de um item de ``/api/tags`` (0 se desconhecido)."""
    texto = str((item.get("details") or {}).get("parameter_size") or "")
    encontrado = re.search(r"(\d+(?:\.\d+)?)\s*([kmb])", texto, re.IGNORECASE)
    if not encontrado:
        return 0.0
    fator = {"k": 1e3, "m": 1e6, "b": 1e9}[encontrado.group(2).lower()]
    return float(encontrado.group(1)) * fator


def _tags(host: str) -> list[dict[str, Any]]:
    """Metadados dos modelos instalados; falha de rede nunca derruba a rodada."""
    try:
        resposta = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=5)
        if resposta.status_code >= 400:
            return []
        dados = resposta.json()
        return [m for m in dados.get("models", []) if isinstance(m, dict)]
    except Exception:
        return []


def maior_modelo_registrado() -> str:
    """Tag do maior modelo ativo no catálogo do load balancer.

    O catálogo não guarda tamanho: as tags registradas são cruzadas com
    ``/api/tags`` do Ollama de cada modelo (local ou remoto). Modelos que não
    estejam instalados são ignorados. Sem candidatos, devolve string vazia.
    """
    try:
        linhas = _catalogo()
    except Exception:
        return ""
    ativos = [linha for linha in linhas if str(linha.get("status") or "").lower() == "up"]
    candidatos = ativos or linhas

    por_host: dict[str, list[dict[str, Any]]] = {}
    melhor: tuple[float, float] = (-1.0, -1.0)
    escolhido = ""
    for linha in candidatos:
        nome = str(linha.get("name") or "").strip()
        if not nome:
            continue
        host = str(linha.get("url") or OLLAMA_HOST).strip().rstrip("/") or OLLAMA_HOST
        if host not in por_host:
            por_host[host] = _tags(host)
        instalado = next(
            (m for m in por_host[host] if _mesmo_modelo(str(m.get("name") or ""), nome)),
            None,
        )
        if instalado is None:
            continue
        tamanho = (_parametros(instalado), float(instalado.get("size") or 0))
        if tamanho > melhor:
            # Mantém a tag canônica do catálogo (ex.: `phi3.5`, não `phi3.5:latest`).
            melhor, escolhido = tamanho, nome
    return escolhido


def modelo_padrao() -> str:
    """Modelo usado quando a rodada não informa ``--model-tag``."""
    return (
        os.getenv("MODELO_BASELINE", "").strip()
        or maior_modelo_registrado()
        or MODELO_FALLBACK
    )
