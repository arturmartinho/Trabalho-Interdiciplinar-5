"""API HTTP do load balancer."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.loadbalancer.dispatcher import dispatch, resolve_classifier
from src.loadbalancer.ollama_client import OllamaClient
from src.loadbalancer.schemas import ClassifierError
from src.loadbalancer.settings import (
    API_HOST,
    API_PORT,
    FALLBACK_MODEL,
    MODEL_CODE,
    MODEL_REASONING,
    MODEL_SIMPLE,
    OLLAMA_HOST,
    OLLAMA_TIMEOUT_S,
)
from src.log import error


class InferenceRequest(BaseModel):
    prompt: str = Field(min_length=1)
    prompt_id: str | None = None


class ChatRequest(BaseModel):
    """Formato de conversa. O campo ``model`` é ignorado: quem escolhe é o classificador."""

    messages: list[dict]
    model: str | None = None
    prompt_id: str | None = None


def create_app(client: OllamaClient | None = None, classifier=None) -> FastAPI:
    """Monta a aplicação. Cliente e classificador podem ser injetados nos testes."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.client is None:
            app.state.client = OllamaClient(OLLAMA_HOST, OLLAMA_TIMEOUT_S)
        try:
            yield
        finally:
            if app.state.owns_client and app.state.client is not None:
                await app.state.client.aclose()

    app = FastAPI(
        title="Load Balancer de modelos de IA",
        lifespan=lifespan,
    )
    app.state.client = client
    app.state.classifier = classifier
    app.state.owns_client = client is None

    @app.get("/health")
    async def health() -> JSONResponse:
        ok, models, problem = await app.state.client.list_models()
        host = getattr(app.state.client, "base_url", OLLAMA_HOST)
        body = {
            "ok": ok,
            "ollama_host": host,
            "fallback_model": FALLBACK_MODEL,
            "expected_models": [MODEL_SIMPLE, MODEL_CODE, MODEL_REASONING],
            "models": models,
            "error": problem,
        }
        return JSONResponse(status_code=200 if ok else 503, content=body)

    @app.post("/infer")
    async def infer(body: InferenceRequest) -> JSONResponse:
        return await _run(app, body.prompt, body.prompt_id)

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatRequest) -> JSONResponse:
        prompt = _last_user_message(body.messages)
        if prompt is None:
            return JSONResponse(
                status_code=422,
                content={"ok": False, "error": "nenhuma mensagem de usuário em messages"},
            )
        return await _run(app, prompt, body.prompt_id)

    return app


async def _run(app: FastAPI, prompt: str, prompt_id: str | None) -> JSONResponse:
    try:
        classifier = _classifier_of(app)
    except ClassifierError as exc:
        error(str(exc))
        return JSONResponse(status_code=503, content={"ok": False, "error": str(exc)})

    try:
        result = await dispatch(
            prompt,
            classifier=classifier,
            client=app.state.client,
            prompt_id=prompt_id,
            fallback_model=FALLBACK_MODEL,
        )
    except ClassifierError as exc:
        error(f"classificador rejeitou o prompt: {exc}")
        return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})

    status = 200 if result.ok else 502
    return JSONResponse(status_code=status, content=result.as_dict())


def _classifier_of(app: FastAPI):
    if app.state.classifier is None:
        app.state.classifier = resolve_classifier()
    return app.state.classifier


def _last_user_message(messages: list[dict]) -> str | None:
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return None


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "src.loadbalancer.api:app",
        host=API_HOST,
        port=API_PORT,
    )
