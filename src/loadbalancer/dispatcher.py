"""Despacho: classifica o prompt, chama o Ollama e aplica o fallback."""

import inspect
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from src.loadbalancer.schemas import Classification, ClassifierError, DispatchResult
from src.loadbalancer.settings import FALLBACK_MODEL
from src.log import error, info, warning

Classifier = Callable[[str], Any | Awaitable[Any]]


def same_model(left: str, right: str) -> bool:
    """Compara a família do modelo, ignorando o tag depois dos dois-pontos."""
    return left.split(":", 1)[0].strip().lower() == right.split(":", 1)[0].strip().lower()


async def dispatch(
    prompt: str,
    *,
    classifier: Classifier,
    client: Any,
    prompt_id: str | None = None,
    fallback_model: str = FALLBACK_MODEL,
    request_id: str | None = None,
) -> DispatchResult:
    """Encaminha um prompt e devolve latências, modelo e eventual desvio.

    Se o modelo escolhido não responder, o mesmo prompt vai para
    ``fallback_model`` (por padrão o Phi-3.5-mini). Não há segunda
    tentativa quando o escolhido já é o modelo de reserva.
    """
    started = time.perf_counter()
    request_id = request_id or uuid.uuid4().hex[:12]
    classification = await _classify(classifier, prompt, prompt_id)

    info(
        "despacho "
        f"request_id={request_id} "
        f"prompt_id={prompt_id or '-'} "
        f"classe={classification.label} "
        f"modelo={classification.model} "
        f"motivo={classification.reason or '-'}"
    )

    first = await client.generate(classification.model, prompt)
    redirected = False
    redirect_reason: str | None = None
    served_model = classification.model
    final = first

    if not first.ok and not same_model(classification.model, fallback_model):
        redirected = True
        redirect_reason = first.error or "sem resposta"
        warning(
            "redirecionamento "
            f"request_id={request_id} "
            f"modelo_escolhido={classification.model} "
            f"fallback={fallback_model} "
            f"motivo={redirect_reason}"
        )
        final = await client.generate(fallback_model, prompt)
        served_model = fallback_model
    elif not first.ok:
        warning(
            "sem redirecionamento "
            f"request_id={request_id} "
            f"modelo={classification.model} "
            f"motivo={first.error or 'sem resposta'} "
            "ja_e_o_fallback=true"
        )

    e2e_latency_s = round(time.perf_counter() - started, 6)
    result = DispatchResult(
        request_id=request_id,
        prompt_id=prompt_id,
        response=final.text if final.ok else None,
        label=classification.label,
        reason=classification.reason,
        chosen_model=classification.model,
        served_model=served_model,
        redirected=redirected,
        redirect_reason=redirect_reason,
        attempt_latency_s=first.latency_s,
        ollama_latency_s=final.latency_s,
        e2e_latency_s=e2e_latency_s,
        ok=final.ok,
        error=None if final.ok else final.error,
    )
    _log_result(result)
    return result


async def _classify(
    classifier: Classifier,
    prompt: str,
    prompt_id: str | None,
) -> Classification:
    try:
        raw = _call_classifier(classifier, prompt, prompt_id)
        if inspect.isawaitable(raw):
            raw = await raw
        return Classification.from_value(raw)
    except ClassifierError:
        raise
    except Exception as exc:
        error(f"classificador falhou: {exc}")
        raise ClassifierError(str(exc)) from exc


def _log_result(result: DispatchResult) -> None:
    message = (
        "resultado "
        f"request_id={result.request_id} "
        f"classe={result.label} "
        f"modelo_escolhido={result.chosen_model} "
        f"modelo_atendido={result.served_model} "
        f"redirecionado={str(result.redirected).lower()} "
        f"latencia_tentativa_s={result.attempt_latency_s:.6f} "
        f"latencia_ollama_s={result.ollama_latency_s:.6f} "
        f"latencia_total_s={result.e2e_latency_s:.6f} "
        f"ok={str(result.ok).lower()}"
    )
    if result.ok:
        info(message)
    else:
        error(f"{message} erro={result.error}")


def _call_classifier(classifier: Classifier, prompt: str, prompt_id: str | None) -> Any:
    """Repassa o id quando a função é ``classificar_e_registrar``."""
    try:
        parameters = inspect.signature(classifier).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "prompt_id" in parameters:
        return classifier(prompt, prompt_id=prompt_id)
    return classifier(prompt)


def resolve_classifier() -> Classifier:
    """Importa o classificador de ``src/classificador.py``.

    A função usada no despacho é ``classificar_e_registrar``, que grava
    classe, modelo e motivo. O módulo dela importa ``log`` como nome solto,
    então a pasta ``src`` entra no caminho de importação.
    """
    src_dir = str(Path(__file__).resolve().parents[1])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    try:
        from classificador import classificar_e_registrar
    except ImportError as exc:
        raise ClassifierError(
            "Classificador não encontrado. "
            "Esperado: função classificar(prompt) em src/classificador.py."
        ) from exc
    return classificar_e_registrar
