"""Normalização de texto e contagem de tokens usadas pelo classificador.

A contagem em tokens substitui a contagem em palavras: o custo de um prompt para o
modelo (tempo de prefill, limite de contexto) cresce com tokens, e a razão palavras/tokens
muda muito entre idiomas e entre texto e código.

Tokenizador (variável ``TOKENIZADOR`` no .env):
  - id de um repositório do Hugging Face (ex.: ``microsoft/Phi-3.5-mini-instruct``), baixado
    uma vez e lido do cache local nas próximas execuções;
  - caminho para um ``tokenizer.json``;
  - ``estimativa`` (padrão): aproximação por caracteres, sem dependências. Foi calibrada
    nos 40 prompts manuais com o tokenizador do Phi-3.5 (~0,34 token por caractere).

Se o tokenizador configurado não puder ser carregado (sem rede, pacote ``tokenizers``
ausente), a contagem cai para a estimativa e um aviso é registrado uma única vez.
"""

from __future__ import annotations

import math
import os
import unicodedata
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from . import log

load_dotenv()

ESTIMATIVA = "estimativa"
TOKENS_POR_CARACTERE = 0.343  # média dos 40 prompts manuais no tokenizador do Phi-3.5

# Valor lido do ambiente; os testes podem sobrescrever com ``configurar_tokenizador``.
_tokenizador_nome = (os.getenv("TOKENIZADOR") or ESTIMATIVA).strip()


def normalizar(texto: str) -> str:
    """Minúsculas e sem acentos, para casar 'Função' com 'funcao'."""
    sem_acento = unicodedata.normalize("NFKD", texto)
    sem_acento = "".join(c for c in sem_acento if not unicodedata.combining(c))
    return sem_acento.lower()


def contar_palavras(texto: str) -> int:
    return len(texto.split())


def estimar_tokens(texto: str) -> int:
    if not texto or not texto.strip():
        return 0
    return max(1, math.ceil(len(texto) * TOKENS_POR_CARACTERE))


def configurar_tokenizador(nome: str) -> None:
    """Troca o tokenizador em tempo de execução (usado em testes e na avaliação)."""
    global _tokenizador_nome
    _tokenizador_nome = (nome or ESTIMATIVA).strip()


@lru_cache(maxsize=4)
def _carregar(nome: str):
    if nome == ESTIMATIVA:
        return None
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        from tokenizers import Tokenizer
    except ImportError:
        log.warning("pacote 'tokenizers' ausente; contando tokens por estimativa. pip install tokenizers")
        return None
    try:
        if Path(nome).is_file():
            return Tokenizer.from_file(nome)
        return Tokenizer.from_pretrained(nome)
    except Exception as exc:  # sem rede, repositório inexistente etc.
        log.warning(f"nao foi possivel carregar o tokenizador '{nome}' ({exc}); usando estimativa")
        return None


def tokenizador_ativo() -> str:
    """Nome do tokenizador efetivamente em uso (``estimativa`` se o configurado falhou)."""
    return _tokenizador_nome if _carregar(_tokenizador_nome) is not None else ESTIMATIVA


def contar_tokens(texto: str) -> int:
    if not texto or not texto.strip():
        return 0
    tok = _carregar(_tokenizador_nome)
    if tok is None:
        return estimar_tokens(texto)
    return len(tok.encode(texto, add_special_tokens=False).ids)
