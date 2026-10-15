"""Configuração do núcleo do load balancer.

Os nomes dos modelos precisam ser os mesmos tags que o Ollama lista
em ``ollama list``. O classificador deve devolver um desses nomes.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_any(names: tuple[str, ...], default: str) -> str:
    """Lê o primeiro nome definido. Aceita os nomes do classificador e os antigos."""
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default


OLLAMA_HOST = _env("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_TIMEOUT_S = float(_env("OLLAMA_TIMEOUT_SECONDS", "120"))

# Mesmos nomes e tags de src/classificador.py (branch feat/classificador).
MODEL_SIMPLE = _env_any(("MODELO_SIMPLES", "MODEL_SIMPLE"), "ailo-152m-v2")
MODEL_CODE = _env_any(("MODELO_CODIGO", "MODEL_CODE"), "qwen2.5-coder:0.5b")
MODEL_REASONING = _env_any(("MODELO_COMPLEXO", "MODEL_REASONING"), "phi3.5")
FALLBACK_MODEL = _env("FALLBACK_MODEL", MODEL_REASONING)

API_HOST = _env("LOADBALANCER_HOST", "127.0.0.1")
API_PORT = int(_env("LOADBALANCER_PORT", "8000"))
