"""Classificador de prompts por regras.

Decide para qual modelo uma requisição deve ir:

- pedido curto e factual           -> AILO-152M-v2       (classe "simples")
- pedido de código                 -> Qwen2.5-Coder-0.5B (classe "codigo")
- pedido longo ou de raciocínio    -> Phi-3.5-mini       (classe "complexo")

`classificar()` é uma função pura: mesma entrada, mesma saída, sem I/O.
`classificar_e_registrar()` chama `classificar()` e grava a decisão no log,
para que cada latência medida tenha a classe, o modelo e o motivo associados.
"""

import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

import log

load_dotenv()

# --------------------------------------------------------------------------- #
# Modelos (tags do Ollama; podem ser sobrescritas no .env)
# --------------------------------------------------------------------------- #
MODELO_SIMPLES = os.getenv("MODELO_SIMPLES", "ailo-152m-v2")
MODELO_CODIGO = os.getenv("MODELO_CODIGO", "qwen2.5-coder:0.5b")
MODELO_COMPLEXO = os.getenv("MODELO_COMPLEXO", "phi3.5")

CLASSE_SIMPLES = "simples"
CLASSE_CODIGO = "codigo"
CLASSE_COMPLEXO = "complexo"

MODELO_POR_CLASSE = {
    CLASSE_SIMPLES: MODELO_SIMPLES,
    CLASSE_CODIGO: MODELO_CODIGO,
    CLASSE_COMPLEXO: MODELO_COMPLEXO,
}

# --------------------------------------------------------------------------- #
# Limiares (em palavras)
# --------------------------------------------------------------------------- #
LIMITE_CURTO = 25    # até aqui, sem outros sinais, o pedido é "simples"
LIMITE_LONGO = 120   # acima disso, o pedido é "complexo" independente do tema

# --------------------------------------------------------------------------- #
# Sinais (comparados com o texto em minúsculas e sem acentos)
# --------------------------------------------------------------------------- #
SINAIS_CODIGO: List[str] = [
    r"```",
    r"\bcodigo\b", r"\bcode\b", r"\bfuncao\b", r"\bfunction\b", r"\bmetodo\b",
    r"\bclasse\b", r"\bscript\b", r"\bprograma\b", r"\bprogram\b",
    r"\bimplemente\b", r"\bimplementar\b", r"\bimplement\b", r"\brefator\w*",
    r"\bdebug\w*", r"\bbug\b", r"\bcompila\w*", r"\bstack ?trace\b",
    r"\bregex\b", r"\bapi\b", r"\bendpoint\b",
    r"\bpython\b", r"\bjava\b", r"\bjavascript\b", r"\btypescript\b",
    r"\bc\+\+", r"\bc#", r"\brust\b", r"\bgolang\b", r"\bsql\b", r"\bhtml\b",
    r"\bcss\b", r"\bbash\b", r"\bpowershell\b",
    r"\bdef \w+\(", r"\bclass \w+", r"\bimport \w+", r"\breturn\b",
    r"\bprint\(", r"\bconsole\.log\b", r"\bselect .+ from\b",
]

SINAIS_RACIOCINIO: List[str] = [
    r"\bexplique\b", r"\bexplicar\b", r"\bexplain\b",
    r"\bpor que\b", r"\bporque\b", r"\bpor qual motivo\b", r"\bwhy\b",
    r"\bcompare\b", r"\bcomparar\b", r"\bcompara\w*",
    r"\banalise\b", r"\banalisar\b", r"\banalyze\b",
    r"\bjustifique\b", r"\bdemonstre\b", r"\bprove\b",
    r"\bpasso a passo\b", r"\bstep by step\b",
    r"\bvantagens e desvantagens\b", r"\bpros e contras\b",
    r"\bresolva\b", r"\bsolve\b", r"\bcalcule\b",
    r"\bavalie\b", r"\bargumente\b", r"\bdiscuta\b", r"\braciocin\w*",
]

_RE_CODIGO = [re.compile(p) for p in SINAIS_CODIGO]
_RE_RACIOCINIO = [re.compile(p) for p in SINAIS_RACIOCINIO]


@dataclass(frozen=True)
class Classificacao:
    """Resultado da classificação de um prompt."""

    classe: str
    modelo: str
    motivo: str
    palavras: int

    def como_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Funções auxiliares
# --------------------------------------------------------------------------- #
def _normalizar(texto: str) -> str:
    """Minúsculas e sem acentos, para casar 'Função' com 'funcao'."""
    sem_acento = unicodedata.normalize("NFKD", texto)
    sem_acento = "".join(c for c in sem_acento if not unicodedata.combining(c))
    return sem_acento.lower()


def _contar_palavras(texto: str) -> int:
    return len(texto.split())


def _sinais_encontrados(texto: str, padroes: List[re.Pattern]) -> List[str]:
    achados = []
    for padrao in padroes:
        m = padrao.search(texto)
        if m:
            achados.append(m.group(0).strip())
    return achados


def _resultado(classe: str, motivo: str, palavras: int) -> Classificacao:
    return Classificacao(classe, MODELO_POR_CLASSE[classe], motivo, palavras)


# --------------------------------------------------------------------------- #
# Classificador (função pura)
# --------------------------------------------------------------------------- #
def classificar(prompt: str) -> Classificacao:
    """Classifica o prompt e escolhe o modelo.

    Ordem das regras (a primeira que casar decide):
      1. prompt vazio                       -> simples
      2. mais de LIMITE_LONGO palavras      -> complexo (contexto longo)
      3. sinal de código                    -> codigo
      4. sinal de raciocínio                -> complexo
      5. até LIMITE_CURTO palavras          -> simples
      6. nenhum sinal e tamanho médio       -> complexo (padrão conservador)
    """
    texto = _normalizar(prompt or "")
    palavras = _contar_palavras(texto)

    if palavras == 0:
        return _resultado(CLASSE_SIMPLES, "prompt vazio", palavras)

    if palavras > LIMITE_LONGO:
        return _resultado(
            CLASSE_COMPLEXO,
            f"pedido longo ({palavras} palavras > {LIMITE_LONGO})",
            palavras,
        )

    codigo = _sinais_encontrados(texto, _RE_CODIGO)
    raciocinio = _sinais_encontrados(texto, _RE_RACIOCINIO)

    if codigo:
        motivo = f"sinais de codigo: {', '.join(codigo[:3])}"
        if raciocinio:
            motivo += f" (tambem raciocinio: {', '.join(raciocinio[:3])}; codigo tem prioridade)"
        return _resultado(CLASSE_CODIGO, motivo, palavras)

    if raciocinio:
        return _resultado(
            CLASSE_COMPLEXO,
            f"sinais de raciocinio: {', '.join(raciocinio[:3])}",
            palavras,
        )

    if palavras <= LIMITE_CURTO:
        return _resultado(
            CLASSE_SIMPLES,
            f"pedido curto ({palavras} palavras <= {LIMITE_CURTO}) sem sinais de codigo/raciocinio",
            palavras,
        )

    return _resultado(
        CLASSE_COMPLEXO,
        f"ambiguo: {palavras} palavras, sem sinais; padrao conservador",
        palavras,
    )


# --------------------------------------------------------------------------- #
# Classificação com registro em log
# --------------------------------------------------------------------------- #
def classificar_e_registrar(prompt: str, prompt_id: Optional[str] = None) -> Classificacao:
    """Classifica o prompt e registra classe, modelo e motivo no log."""
    resultado = classificar(prompt)
    ident = f"id={prompt_id} " if prompt_id is not None else ""
    log.info(
        f"[classificador] {ident}classe={resultado.classe} "
        f"modelo={resultado.modelo} palavras={resultado.palavras} "
        f"motivo=\"{resultado.motivo}\""
    )
    return resultado
