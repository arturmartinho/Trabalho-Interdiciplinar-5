"""Classificador de prompts por regras.

Decide para qual modelo uma requisição deve ir:

- pedido curto e factual           -> AILO-152M-v2       (classe "simples")
- pedido de código                 -> Qwen2.5-Coder-0.5B (classe "codigo")
- pedido longo ou de raciocínio    -> Phi-3.5-mini       (classe "complexo")

O nível de raciocínio é medido pela Taxonomia de Bloom revisada (Anderson e
Krathwohl, 2001): o verbo de comando do prompt indica o nível cognitivo
(1 Lembrar, 2 Entender, 3 Aplicar, 4 Analisar, 5 Avaliar, 6 Criar). Pedidos a
partir de LIMIAR_BLOOM vão para o modelo maior.

`classificar()` é uma função pura: mesma entrada, mesma saída, sem I/O.
`classificar_e_registrar()` chama `classificar()` e grava a decisão no log,
para que cada latência medida tenha a classe, o modelo e o motivo associados.
"""

import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

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
# Limiares
# --------------------------------------------------------------------------- #
LIMITE_CURTO = 25    # até aqui, sem outros sinais, o pedido é "simples"
LIMITE_LONGO = 120   # acima disso, o pedido é "complexo" independente do tema

# Nível de Bloom a partir do qual o pedido é "complexo". Com 2, só "Lembrar"
# vai para o modelo pequeno: um modelo de 152M não sustenta explicações.
LIMIAR_BLOOM = int(os.getenv("LIMIAR_BLOOM", "2"))

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

# Verbos de comando por nível da Taxonomia de Bloom revisada.
NIVEIS_BLOOM: Dict[int, str] = {
    1: "Lembrar",
    2: "Entender",
    3: "Aplicar",
    4: "Analisar",
    5: "Avaliar",
    6: "Criar",
}


def _radical(raiz: str) -> str:
    """Palavras que começam com `raiz`, exceto o pretérito ('escreveu', 'explicou').

    O nível de Bloom vem do verbo de comando: "Quem escreveu Dom Casmurro?" é
    Lembrar, não Criar.
    """
    return rf"\b{raiz}(?!\w*(?:eu|ou|iu|ram)\b)\w*"


SINAIS_BLOOM: Dict[int, List[str]] = {
    1: [
        r"\bqual\b", r"\bquais\b", r"\bquem\b", r"\bquando\b", r"\bonde\b",
        r"\bo que e\b", r"\bliste\b", r"\blistar\b", r"\bcite\b", _radical("defin"),
        r"\bnomeie\b", r"\bidentifique\b",
        r"\bwhat is\b", r"\bwho\b", r"\bwhen\b", r"\bwhere\b", r"\blist\b",
        r"\bdefine\b",
    ],
    2: [
        _radical("expli"), _radical("explain"), _radical("descrev"), r"\bdescribe\b",
        _radical("resum"), _radical("summar"), _radical("interpret"), _radical("parafrase"),
        r"\bparaphrase\b", r"\bexemplifique\b", _radical("traduz"), r"\btranslate\b",
        r"\bpor que\b", r"\bporque\b", r"\bpor qual motivo\b", r"\bwhy\b",
        r"\bcomo funciona\b", r"\bhow does\b",
    ],
    3: [
        _radical("resolv"), r"\bsolve\b", _radical("calcul"), r"\baplique\b",
        r"\baplicar\b", r"\bapply\b", r"\bdemonstre\b", r"\bdemonstrate\b",
        r"\bconverta\b", r"\bconvert\b", r"\bestime\b", r"\bestimate\b",
        r"\bpasso a passo\b", r"\bstep by step\b",
        r"\bcomo (fazer|faco|faz|posso|devo)\b", r"\bhow (to|do i|can i)\b",
        r"\d+\s*[-+*/^x]\s*\d+", r"\d+\s*%",
    ],
    4: [
        _radical("compar"), _radical("analis"), _radical("analy"), _radical("diferenc"),
        r"\bdifference\b", _radical("contrast"), _radical("relacion"), r"\binvestigue\b",
        r"\bexamine\b", r"\bvantagens e desvantagens\b", r"\bpros e contras\b",
        r"\bpros and cons\b",
    ],
    5: [
        _radical("avali"), _radical("evaluat"), _radical("justifi"), r"\bjustify\b",
        _radical("argument"), _radical("critiq"), r"\bdefenda\b", r"\bjulgue\b",
        _radical("recomend"), _radical("recommend"), _radical("discut"), _radical("discuss"),
        r"\bqual (e )?(o|a) melhor\b", r"\bvale a pena\b",
    ],
    6: [
        r"\bcrie\b", r"\bcriar\b", r"\bcreate\b", _radical("elabor"), _radical("redij"),
        r"\bredigir\b", _radical("escrev"), r"\bwrite\b", r"\bcomponha\b",
        r"\bcompose\b", r"\bproponha\b", r"\bpropose\b", _radical("planej"),
        r"\bplan\b", r"\bprojete\b", r"\binvente\b", r"\bgere\b",
        r"\bgenerate\b", _radical("sugir"), _radical("sugest"), _radical("suggest"),
    ],
}

_RE_CODIGO = [re.compile(p) for p in SINAIS_CODIGO]
_RE_BLOOM = {nivel: [re.compile(p) for p in padroes] for nivel, padroes in SINAIS_BLOOM.items()}


@dataclass(frozen=True)
class Classificacao:
    """Resultado da classificação de um prompt."""

    classe: str
    modelo: str
    motivo: str
    palavras: int
    nivel_bloom: int

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


def _nivel_bloom(texto: str) -> Tuple[int, List[str]]:
    """Maior nível de Bloom com sinal no texto já normalizado (1 se nenhum)."""
    for nivel in sorted(_RE_BLOOM, reverse=True):
        achados = _sinais_encontrados(texto, _RE_BLOOM[nivel])
        if achados:
            return nivel, achados
    return 1, []


def _descrever_bloom(nivel: int, sinais: List[str]) -> str:
    descricao = f"Bloom {nivel} ({NIVEIS_BLOOM[nivel]})"
    if sinais:
        descricao += f": {', '.join(sinais[:3])}"
    return descricao


def _resultado(classe: str, motivo: str, palavras: int, nivel: int) -> Classificacao:
    return Classificacao(classe, MODELO_POR_CLASSE[classe], motivo, palavras, nivel)


def detectar_nivel_bloom(prompt: str) -> Tuple[int, str]:
    """Nível de Bloom do prompt e o nome do nível, ex.: (4, "Analisar")."""
    nivel, _ = _nivel_bloom(_normalizar(prompt or ""))
    return nivel, NIVEIS_BLOOM[nivel]


# --------------------------------------------------------------------------- #
# Classificador (função pura)
# --------------------------------------------------------------------------- #
def classificar(prompt: str) -> Classificacao:
    """Classifica o prompt e escolhe o modelo.

    Ordem das regras (a primeira que casar decide):
      1. prompt vazio                         -> simples
      2. mais de LIMITE_LONGO palavras        -> complexo (contexto longo)
      3. sinal de código                      -> codigo
      4. nível de Bloom >= LIMIAR_BLOOM       -> complexo
      5. até LIMITE_CURTO palavras            -> simples
      6. nenhum sinal e tamanho médio         -> complexo (padrão conservador)
    """
    texto = _normalizar(prompt or "")
    palavras = _contar_palavras(texto)

    if palavras == 0:
        return _resultado(CLASSE_SIMPLES, "prompt vazio", palavras, 1)

    nivel, sinais_bloom = _nivel_bloom(texto)
    bloom = _descrever_bloom(nivel, sinais_bloom)

    if palavras > LIMITE_LONGO:
        return _resultado(
            CLASSE_COMPLEXO,
            f"pedido longo ({palavras} palavras > {LIMITE_LONGO}); {bloom}",
            palavras,
            nivel,
        )

    codigo = _sinais_encontrados(texto, _RE_CODIGO)

    if codigo:
        motivo = f"sinais de codigo: {', '.join(codigo[:3])}"
        if nivel >= LIMIAR_BLOOM:
            motivo += f" (tambem {bloom}; codigo tem prioridade)"
        return _resultado(CLASSE_CODIGO, motivo, palavras, nivel)

    if nivel >= LIMIAR_BLOOM:
        return _resultado(
            CLASSE_COMPLEXO,
            f"{bloom} >= limiar {LIMIAR_BLOOM}",
            palavras,
            nivel,
        )

    if palavras <= LIMITE_CURTO:
        return _resultado(
            CLASSE_SIMPLES,
            f"{bloom}, pedido curto ({palavras} palavras <= {LIMITE_CURTO}) sem sinais de codigo",
            palavras,
            nivel,
        )

    return _resultado(
        CLASSE_COMPLEXO,
        f"ambiguo: {palavras} palavras, {bloom}, sem sinais de codigo; padrao conservador",
        palavras,
        nivel,
    )


# --------------------------------------------------------------------------- #
# Classificação com registro em log
# --------------------------------------------------------------------------- #
def classificar_e_registrar(prompt: str, prompt_id: Optional[str] = None) -> Classificacao:
    """Classifica o prompt e registra classe, modelo, nível de Bloom e motivo no log."""
    resultado = classificar(prompt)
    ident = f"id={prompt_id} " if prompt_id is not None else ""
    log.info(
        f"[classificador] {ident}classe={resultado.classe} "
        f"modelo={resultado.modelo} bloom={resultado.nivel_bloom} "
        f"palavras={resultado.palavras} motivo=\"{resultado.motivo}\""
    )
    return resultado
