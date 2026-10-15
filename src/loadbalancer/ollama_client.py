"""Cliente HTTP do Ollama, com timeout e latência de ponta a ponta."""

import os
import time

import httpx

from src.loadbalancer.schemas import OllamaCall


class OllamaClient:
    """Envia um prompt a ``POST /api/generate`` e mede a chamada inteira.

    A latência começa imediatamente antes do envio e termina quando a
    resposta chega ou quando a chamada falha. Esse intervalo inclui a
    carga do modelo, se o Ollama ainda não o tiver em memória.
    """

    def __init__(
        self,
        base_url: str,
        timeout_s: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate(self, model: str, prompt: str) -> OllamaCall:
        """Chama o modelo. Qualquer falha vira ``ok=False``, sem exceção."""
        started = time.perf_counter()
        try:
            response = await self._client.post(
                "/api/generate",
                json=_corpo_da_geracao(model, prompt),
            )
        except httpx.TimeoutException as exc:
            return OllamaCall(
                ok=False,
                latency_s=_elapsed(started),
                error=f"timeout: {exc}",
            )
        except httpx.HTTPError as exc:
            return OllamaCall(
                ok=False,
                latency_s=_elapsed(started),
                error=f"conexao: {exc}",
            )

        latency_s = _elapsed(started)
        if response.status_code >= 400:
            detail = response.text.strip().replace("\n", " ")
            if len(detail) > 300:
                detail = detail[:300] + "..."
            return OllamaCall(
                ok=False,
                latency_s=latency_s,
                error=f"http {response.status_code}: {detail}",
                status_code=response.status_code,
            )

        try:
            payload = response.json()
        except ValueError:
            return OllamaCall(
                ok=False,
                latency_s=latency_s,
                error="corpo da resposta não é JSON",
                status_code=response.status_code,
            )

        if not isinstance(payload, dict):
            return OllamaCall(
                ok=False,
                latency_s=latency_s,
                error="corpo da resposta não é um objeto JSON",
                status_code=response.status_code,
            )

        if payload.get("error"):
            return OllamaCall(
                ok=False,
                latency_s=latency_s,
                error=str(payload["error"]),
                status_code=response.status_code,
            )

        text = str(payload.get("response") or "").strip()
        if not text:
            return OllamaCall(
                ok=False,
                latency_s=latency_s,
                error="resposta vazia",
                status_code=response.status_code,
            )

        return OllamaCall(
            ok=True,
            latency_s=latency_s,
            text=text,
            status_code=response.status_code,
        )

    async def list_models(self) -> tuple[bool, list[str], str | None]:
        """Consulta ``GET /api/tags``. Usado só para checar o ambiente."""
        try:
            response = await self._client.get("/api/tags", timeout=5.0)
        except httpx.HTTPError as exc:
            return False, [], str(exc)
        if response.status_code >= 400:
            return False, [], f"http {response.status_code}"
        try:
            payload = response.json()
        except ValueError:
            return False, [], "corpo da resposta não é JSON"
        names = [
            str(item.get("name"))
            for item in payload.get("models", [])
            if isinstance(item, dict) and item.get("name")
        ]
        return True, names, None


def _corpo_da_geracao(model: str, prompt: str) -> dict:
    """Monta o pedido. ``OLLAMA_NUM_PREDICT`` limita o tamanho só quando está definido."""
    corpo: dict = {"model": model, "prompt": prompt, "stream": False}
    limite = os.getenv("OLLAMA_NUM_PREDICT", "").strip()
    if limite:
        corpo["options"] = {"num_predict": int(limite)}
    return corpo


def _elapsed(started: float) -> float:
    return round(time.perf_counter() - started, 6)
