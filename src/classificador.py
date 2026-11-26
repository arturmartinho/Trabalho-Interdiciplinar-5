"""Classificador de prompts.

Decide para qual modelo uma requisição deve ir:

- pedido curto e factual           -> Qwen2.5 1.5B       (classe "simples")
- pedido de código                 -> Qwen2.5-Coder-1.5B (classe "codigo")
- pedido longo ou de raciocínio    -> Phi-3.5-mini       (classe "complexo")

Três estratégias, escolhidas por ``CLASSIFICADOR`` no .env (ou pelo argumento
``estrategia``):

- ``regras``  : regras determinísticas (tamanho em tokens + sinais por regex).
- ``ml``      : modelo treinado (TF-IDF ou embeddings + Regressão Logística).
- ``hibrido`` : sinais fortes de código decidem direto; o modelo treinado decide o
                resto; se a confiança dele ficar abaixo de ``CLASSIFICADOR_LIMIAR``,
                valem as regras. Prompts longos ou com sinal explícito de raciocínio
                nunca vão para o modelo simples.

Se o modelo treinado não existir ou o scikit-learn não estiver instalado, ``ml`` e
``hibrido`` usam as regras e dizem isso no motivo.

Com ``regras`` (e com ``ml``/``hibrido`` sobre TF-IDF), ``classificar()`` é uma função
pura: mesma entrada, mesma saída, sem I/O. Com embeddings, há uma chamada ao Ollama.
``classificar_e_registrar()`` chama ``classificar()`` e grava a decisão no log.
"""

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from . import log
from .texto import contar_palavras, contar_tokens, normalizar, tokenizador_ativo

load_dotenv()

_RAIZ = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- #
# Modelos (tags do Ollama; podem ser sobrescritas no .env)
# --------------------------------------------------------------------------- #
MODELO_SIMPLES = os.getenv("MODELO_SIMPLES", "qwen2.5-simples:1.5b")
MODELO_CODIGO = os.getenv("MODELO_CODIGO", "qwen2.5-coder:1.5b")
MODELO_COMPLEXO = os.getenv("MODELO_COMPLEXO", "phi3.5")

CLASSE_SIMPLES = "simples"
CLASSE_CODIGO = "codigo"
CLASSE_COMPLEXO = "complexo"
CLASSES = (CLASSE_SIMPLES, CLASSE_CODIGO, CLASSE_COMPLEXO)

MODELO_POR_CLASSE = {
    CLASSE_SIMPLES: MODELO_SIMPLES,
    CLASSE_CODIGO: MODELO_CODIGO,
    CLASSE_COMPLEXO: MODELO_COMPLEXO,
}

# --------------------------------------------------------------------------- #
# Estratégia
# --------------------------------------------------------------------------- #
ESTRATEGIAS = ("regras", "ml", "hibrido")
ESTRATEGIA = (os.getenv("CLASSIFICADOR") or "regras").strip().lower()
CAMINHO_MODELO_ML = str(_RAIZ / (os.getenv("CLASSIFICADOR_MODELO") or "modelos/classificador_tfidf.joblib"))
# (um caminho absoluto no .env é mantido; um relativo é lido a partir da raiz do repositório)
LIMIAR_CONFIANCA = float(os.getenv("CLASSIFICADOR_LIMIAR") or 0.55)

# --------------------------------------------------------------------------- #
# Limiares em tokens (tokenizador do Phi-3.5; ~2 tokens por palavra em português).
# Equivalem aos antigos 25 e 120 palavras.
# --------------------------------------------------------------------------- #
LIMITE_CURTO = 50    # até aqui, sem outros sinais, o pedido é "simples"
LIMITE_LONGO = 240   # acima disso, o pedido é "complexo" independente do tema

# Limiares antigos, em palavras (mantidos para comparação na avaliação).
LIMITE_CURTO_PALAVRAS = 25
LIMITE_LONGO_PALAVRAS = 120

# --------------------------------------------------------------------------- #
# Sinais (comparados com o texto em minúsculas e sem acentos)
# --------------------------------------------------------------------------- #
# Evitar termos ambíguos isolados (programa/TV, api/banco, return/livro, classe/social).
SINAIS_CODIGO: List[str] = [
    r"```",
    r"\bcodigo\b", r"\bcode\b", r"\bfuncao\b", r"\bfunction\b", r"\bmetodo\b",
    r"\bscript\b",
    r"\bprogram(?:ar|acao)\b",
    r"\bprograma\s+(?:em|para|que)\b",
    r"\bprograma\s+de\s+(?:computador|computacao|python|java|software)\b",
    r"\bescreva\s+(?:um\s+)?programa\b",
    r"\bimplemente\b", r"\bimplementar\b", r"\bimplement\b", r"\brefator\w*",
    r"\bdebug\w*", r"\bbug\b", r"\bcompila\w*", r"\bstack ?trace\b",
    r"\bregex\b", r"\bendpoint\b",
    r"\b(?:rest|http|web)\s+api\b", r"\bapi\s+(?:rest|http|web|endpoint)\b",
    r"\bcriar\s+(?:uma\s+)?api\b", r"\bimplemente\s+(?:uma\s+)?api\b",
    r"\bpython\b", r"\bjava\b", r"\bjavascript\b", r"\btypescript\b",
    r"\bc\+\+", r"\bc#", r"\brust\b", r"\bgolang\b", r"\bsql\b", r"\bhtml\b",
    r"\bcss\b", r"\bbash\b", r"\bpowershell\b",
    r"\bdef \w+\(", r"\bclass \w+", r"\bclasse\s+\w+\s*[:\(]",
    r"\bimport \w+",
    # return de código (não "return o livro")
    r"\breturn\s+(?:None|True|False|\d|[\[\(\{\"']|[a-z_][a-z0-9_]*\s*[=\(\[\+\-*/%])",
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
    # espanhol (o texto já vem sem acento: "por qué" -> "por que", "demuéstrame" -> "demuestrame")
    r"\bexplica(?:me)?\b", r"\bpaso a paso\b", r"\bdemuestra\w*", r"\banaliza\b", r"\bresuelve\b",
    r"\bjustifica\b", r"\bargumenta\b",
]

# Sinais "fortes": sintaxe de código de fato, não apenas o assunto. No modo híbrido
# decidem a classe sem consultar o modelo treinado.
SINAIS_CODIGO_FORTES: List[str] = [
    r"```",
    r"\bdef \w+\s*\(", r"\bclass \w+\s*[:({]", r"^\s*import [\w.]+(?:\s+as\s+\w+)?\s*;?\s*$", r"^\s*from [\w.]+ import\b", r"#include\s*<",
    r"\bconsole\.log\(", r"\bselect\b[^\n]+\bfrom\b[^\n]*\bwhere\b", r"\bpublic (?:static )?(?:void|class|int|string|boolean|double|float|long)\b",
    r"\bfn \w+\s*\(", r"\bfunction\s*\w*\s*\(", r"=>\s*\{", r"\bfunc \w+\s*\(",
    r"\b(?:let|const|var) \w+\s*=", r"^\s*\w+\s*\([^)\n]*\)\s*\{\s*$", r"\btraceback \(most recent call",
    r";\s*$",
]

_RE_CODIGO = [re.compile(p) for p in SINAIS_CODIGO]
_RE_RACIOCINIO = [re.compile(p) for p in SINAIS_RACIOCINIO]
_RE_CODIGO_FORTE = [re.compile(p, re.MULTILINE) for p in SINAIS_CODIGO_FORTES]


@dataclass(frozen=True)
class Classificacao:
    """Resultado da classificação de um prompt."""

    classe: str
    modelo: str
    motivo: str
    palavras: int
    tokens: int = 0
    estrategia: str = "regras"
    confianca: Optional[float] = None

    def como_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Funções auxiliares
# --------------------------------------------------------------------------- #
def _normalizar(texto: str) -> str:  # mantido por compatibilidade
    return normalizar(texto)


def _sinais_encontrados(texto: str, padroes: List[re.Pattern]) -> List[str]:
    achados = []
    for padrao in padroes:
        m = padrao.search(texto)
        if m:
            achados.append(m.group(0).strip())
    return achados


def _mapa_modelos(modelos: Optional[Dict[str, str]]) -> Dict[str, str]:
    if modelos:
        return {**MODELO_POR_CLASSE, **{k: v for k, v in modelos.items() if v}}
    return MODELO_POR_CLASSE


def _regras(
    texto_norm: str,
    tamanho: int,
    unidade: str = "tokens",
    limite_curto: int = LIMITE_CURTO,
    limite_longo: int = LIMITE_LONGO,
) -> Tuple[str, str]:
    """Regras determinísticas (a primeira que casar decide).

      1. prompt vazio                       -> simples
      2. mais de ``limite_longo``           -> complexo (contexto longo)
      3. sinal de código                    -> codigo
      4. sinal de raciocínio                -> complexo
      5. até ``limite_curto``               -> simples
      6. nenhum sinal e tamanho médio       -> complexo (padrão conservador)
    """
    if tamanho == 0:
        return CLASSE_SIMPLES, "prompt vazio"
    if tamanho > limite_longo:
        return CLASSE_COMPLEXO, f"pedido longo ({tamanho} {unidade} > {limite_longo})"

    codigo = _sinais_encontrados(texto_norm, _RE_CODIGO)
    raciocinio = _sinais_encontrados(texto_norm, _RE_RACIOCINIO)
    if codigo:
        motivo = f"sinais de codigo: {', '.join(codigo[:3])}"
        if raciocinio:
            motivo += f" (tambem raciocinio: {', '.join(raciocinio[:3])}; codigo tem prioridade)"
        return CLASSE_CODIGO, motivo
    if raciocinio:
        return CLASSE_COMPLEXO, f"sinais de raciocinio: {', '.join(raciocinio[:3])}"
    if tamanho <= limite_curto:
        return CLASSE_SIMPLES, f"pedido curto ({tamanho} {unidade} <= {limite_curto}) sem sinais de codigo/raciocinio"
    return CLASSE_COMPLEXO, f"ambiguo: {tamanho} {unidade}, sem sinais; padrao conservador"


# --------------------------------------------------------------------------- #
# Modelo treinado
# --------------------------------------------------------------------------- #
_avisos_emitidos: set = set()


def _avisar_uma_vez(chave: str, mensagem: str) -> None:
    if chave not in _avisos_emitidos:
        _avisos_emitidos.add(chave)
        log.warning(mensagem)


def modelo_ml(caminho: Optional[str] = None):
    """Devolve o modelo treinado, ou ``None`` se ele não puder ser usado."""
    caminho = caminho or CAMINHO_MODELO_ML
    if not Path(caminho).is_file():
        _avisar_uma_vez(f"sem:{caminho}", f"[classificador] modelo treinado nao encontrado em {caminho}; "
                        "usando regras. Treine com: python -m src train-classifier")
        return None
    try:
        from .classificador_ml import carregar_cache

        return carregar_cache(caminho)
    except Exception as exc:  # sklearn ausente, arquivo incompatível etc.
        _avisar_uma_vez(f"erro:{caminho}", f"[classificador] falha ao carregar {caminho} ({exc}); usando regras")
        return None


def estrategia_atual(estrategia: Optional[str] = None) -> str:
    e = (estrategia or ESTRATEGIA).strip().lower()
    if e not in ESTRATEGIAS:
        _avisar_uma_vez(f"estr:{e}", f"[classificador] estrategia '{e}' desconhecida; usando regras")
        return "regras"
    return e


# --------------------------------------------------------------------------- #
# Classificador
# --------------------------------------------------------------------------- #
def classificar_regras(prompt: str, unidade: str = "tokens") -> Tuple[str, str, int]:
    """Só as regras. ``unidade="palavras"`` reproduz o classificador antigo (25/120 palavras)."""
    texto = normalizar(prompt or "")
    if unidade == "palavras":
        n = contar_palavras(texto)
        classe, motivo = _regras(texto, n, "palavras", LIMITE_CURTO_PALAVRAS, LIMITE_LONGO_PALAVRAS)
    else:
        n = contar_tokens(prompt or "")
        classe, motivo = _regras(texto, n)
    return classe, motivo, n


def classificar(
    prompt: str,
    modelos: Optional[Dict[str, str]] = None,
    estrategia: Optional[str] = None,
    caminho_modelo: Optional[str] = None,
) -> Classificacao:
    """Classifica o prompt e escolhe o modelo.

    ``modelos`` permite substituir os destinos vindos do ambiente pelos
    modelos registrados no banco (classe -> tag do Ollama). ``estrategia``
    sobrescreve ``CLASSIFICADOR`` do .env.
    """
    prompt = prompt or ""
    texto = normalizar(prompt)
    palavras = contar_palavras(texto)
    tokens = contar_tokens(prompt)
    est = estrategia_atual(estrategia)
    mapa = _mapa_modelos(modelos)

    def resultado(classe: str, motivo: str, conf: Optional[float] = None) -> Classificacao:
        return Classificacao(classe, mapa[classe], motivo, palavras, tokens, est, conf)

    classe_r, motivo_r = _regras(texto, tokens)
    if est == "regras" or tokens == 0:
        return resultado(classe_r, motivo_r)

    if est == "hibrido":
        fortes = _sinais_encontrados(texto, _RE_CODIGO_FORTE)
        if fortes:
            return resultado(CLASSE_CODIGO, f"sintaxe de codigo: {', '.join(fortes[:3])}")

    ml = modelo_ml(caminho_modelo)
    if ml is None:
        return resultado(classe_r, f"modelo ML indisponivel; regras: {motivo_r}")

    classe, conf, probs = ml.prever(prompt)
    probs_txt = " ".join(f"{c}={p:.2f}" for c, p in probs.items())
    if est == "ml":
        return resultado(classe, f"ml[{ml.features}] {probs_txt}", conf)

    if conf < LIMIAR_CONFIANCA:
        motivo = f"ml incerto ({classe} p={conf:.2f} < {LIMIAR_CONFIANCA}); regras: {motivo_r}"
        classe = classe_r
    else:
        motivo = f"ml[{ml.features}] {probs_txt}"
    if classe == CLASSE_SIMPLES and tokens > LIMITE_LONGO:
        classe = CLASSE_COMPLEXO
        motivo += f"; {tokens} tokens > {LIMITE_LONGO}: longo demais para o modelo simples"
    elif classe == CLASSE_SIMPLES:
        # Nos dados gerados quase todo pedido complexo é longo, e o modelo aprende "curto = simples"
        # ("Prove que existem infinitos primos." -> simples). Mandar para o modelo menor é o erro caro
        # (compromete a resposta); para o maior, só custa latência. Então sinal explícito de
        # raciocínio não é rebaixado para simples.
        raciocinio = _sinais_encontrados(texto, _RE_RACIOCINIO)
        if raciocinio:
            classe = CLASSE_COMPLEXO
            motivo += f"; sinais de raciocinio ({', '.join(raciocinio[:3])}): nao rebaixa para simples"
    return resultado(classe, motivo, conf)


def preparar(estrategia: Optional[str] = None, aquecer: bool = True) -> Dict[str, Any]:
    """Carrega tokenizador e modelo treinado e descreve a configuração em uso.

    Com ``aquecer=True`` e embeddings, faz uma chamada para carregar o modelo de
    embeddings no Ollama antes da primeira requisição medida.
    """
    contar_tokens("aquecimento")
    est = estrategia_atual(estrategia)
    info: Dict[str, Any] = {"estrategia": est, "tokenizador": tokenizador_ativo(),
                            "limites_tokens": {"curto": LIMITE_CURTO, "longo": LIMITE_LONGO}}
    if est != "regras":
        ml = modelo_ml()
        info["modelo_ml"] = {
            "caminho": CAMINHO_MODELO_ML,
            "carregado": ml is not None,
            "features": getattr(ml, "features", None),
            "limiar_confianca": LIMIAR_CONFIANCA,
            "metadados": {k: v for k, v in (getattr(ml, "metadados", None) or {}).items()
                          if k in ("treinado_utc", "n_treino", "acuracia_validacao", "embed_modelo")},
        }
        if aquecer and ml is not None and ml.features == "embed":
            ml.prever("aquecimento")  # carrega o modelo de embeddings no Ollama
    return info


def usa_rede(estrategia: Optional[str] = None) -> bool:
    """True quando a classificação faz I/O (embeddings no Ollama)."""
    if estrategia_atual(estrategia) == "regras":
        return False
    ml = modelo_ml()
    return ml is not None and ml.features == "embed"


# --------------------------------------------------------------------------- #
# Classificação com registro em log
# --------------------------------------------------------------------------- #
def classificar_e_registrar(
    prompt: str,
    prompt_id: Optional[str] = None,
    modelos: Optional[Dict[str, str]] = None,
    estrategia: Optional[str] = None,
) -> Classificacao:
    """Classifica o prompt e registra classe, modelo e motivo no log."""
    resultado = classificar(prompt, modelos=modelos, estrategia=estrategia)
    ident = f"id={prompt_id} " if prompt_id is not None else ""
    log.info(
        f"[classificador] {ident}classe={resultado.classe} "
        f"modelo={resultado.modelo} estrategia={resultado.estrategia} "
        f"tokens={resultado.tokens} palavras={resultado.palavras} "
        f"motivo=\"{resultado.motivo}\""
    )
    return resultado
