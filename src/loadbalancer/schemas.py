"""Contratos do despacho, independentes do FastAPI."""

from dataclasses import asdict, dataclass
from typing import Any


class ClassifierError(Exception):
    """O classificador faltou ou devolveu um resultado inutilizável."""


@dataclass(frozen=True)
class Classification:
    """Saída esperada do classificador de prompts (Aluno 2)."""

    label: str
    model: str
    reason: str

    @classmethod
    def from_value(cls, raw: Any) -> "Classification":
        """Aceita dict ou objeto com os campos em português ou em inglês.

        Campos reconhecidos: ``label``/``classe``, ``model``/``modelo``,
        ``reason``/``motivo``.
        """
        if isinstance(raw, Classification):
            label, model, reason = raw.label, raw.model, raw.reason
        elif isinstance(raw, dict):
            label = raw.get("label", raw.get("classe", raw.get("class_name")))
            model = raw.get("model", raw.get("modelo"))
            reason = raw.get("reason", raw.get("motivo", ""))
        else:
            label = getattr(raw, "label", None)
            if label is None:
                label = getattr(raw, "classe", None)
            model = getattr(raw, "model", None)
            if model is None:
                model = getattr(raw, "modelo", None)
            reason = getattr(raw, "reason", None)
            if reason is None:
                reason = getattr(raw, "motivo", "")

        if not label or not str(label).strip():
            raise ClassifierError("o classificador não informou a classe")
        if not model or not str(model).strip():
            raise ClassifierError("o classificador não informou o modelo")
        return cls(str(label).strip(), str(model).strip(), str(reason or "").strip())


@dataclass(frozen=True)
class OllamaCall:
    """Resultado de uma chamada ao Ollama, com a latência medida no cliente."""

    ok: bool
    latency_s: float
    text: str = ""
    error: str | None = None
    status_code: int | None = None


@dataclass(frozen=True)
class DispatchResult:
    """Resultado completo de um despacho, já com fallback se houver."""

    request_id: str
    prompt_id: str | None
    response: str | None
    label: str
    reason: str
    chosen_model: str
    served_model: str
    redirected: bool
    redirect_reason: str | None
    attempt_latency_s: float
    ollama_latency_s: float
    e2e_latency_s: float
    ok: bool
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
