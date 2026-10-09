"""Testes do núcleo do load balancer, sem Ollama real."""

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from src.loadbalancer.api import create_app
from src.loadbalancer.dispatcher import dispatch
from src.loadbalancer.ollama_client import OllamaClient
from src.loadbalancer.schemas import Classification, ClassifierError, OllamaCall
from src.loadbalancer.settings import FALLBACK_MODEL


class FakeClient:
    def __init__(self, results: list[OllamaCall]) -> None:
        self._results = list(results)
        self.calls: list[tuple[str, str]] = []

    async def generate(self, model: str, prompt: str) -> OllamaCall:
        self.calls.append((model, prompt))
        return self._results.pop(0)


def _classify_code(prompt: str) -> dict:
    return {
        "classe": "codigo",
        "modelo": "qwen2.5-coder:0.5b",
        "motivo": "pedido de codigo",
    }


def _run(coro):
    return asyncio.run(coro)


def test_despacho_usa_o_modelo_do_classificador():
    client = FakeClient([OllamaCall(ok=True, latency_s=0.2, text="def soma():\n    return 1")])

    result = _run(
        dispatch(
            "escreva uma função soma",
            classifier=_classify_code,
            client=client,
            prompt_id="p1",
            fallback_model="phi3.5",
            request_id="req1",
        )
    )

    assert client.calls == [("qwen2.5-coder:0.5b", "escreva uma função soma")]
    assert result.ok is True
    assert result.redirected is False
    assert result.chosen_model == "qwen2.5-coder:0.5b"
    assert result.served_model == "qwen2.5-coder:0.5b"
    assert result.label == "codigo"
    assert result.reason == "pedido de codigo"
    assert result.prompt_id == "p1"
    assert result.response.startswith("def soma")
    assert result.attempt_latency_s == 0.2
    assert result.e2e_latency_s >= 0


def test_despacho_redireciona_quando_o_modelo_nao_responde():
    client = FakeClient(
        [
            OllamaCall(ok=False, latency_s=0.1, error="timeout: estouro"),
            OllamaCall(ok=True, latency_s=0.4, text="Paris"),
        ]
    )

    result = _run(
        dispatch(
            "capital da França",
            classifier=lambda _prompt: Classification("simples", "AILO-152M-v2", "pedido curto"),
            client=client,
            fallback_model="phi3.5",
            request_id="req2",
        )
    )

    assert [call[0] for call in client.calls] == ["AILO-152M-v2", "phi3.5"]
    assert result.ok is True
    assert result.redirected is True
    assert result.redirect_reason == "timeout: estouro"
    assert result.served_model == "phi3.5"
    assert result.response == "Paris"
    assert result.attempt_latency_s == 0.1
    assert result.ollama_latency_s == 0.4


def test_despacho_registra_o_redirecionamento_no_log(capsys):
    client = FakeClient(
        [
            OllamaCall(ok=False, latency_s=0.05, error="http 404: model not found"),
            OllamaCall(ok=True, latency_s=0.2, text="ok"),
        ]
    )

    _run(
        dispatch(
            "quanto é 2+2",
            classifier=_classify_code,
            client=client,
            fallback_model="phi3.5",
            request_id="req-log",
        )
    )

    output = capsys.readouterr().out
    assert "redirecionamento" in output
    assert "modelo_escolhido=qwen2.5-coder:0.5b" in output
    assert "fallback=phi3.5" in output
    assert "classe=codigo" in output
    assert "latencia_total_s=" in output


def test_nao_repete_quando_o_escolhido_ja_e_o_fallback():
    client = FakeClient([OllamaCall(ok=False, latency_s=1.5, error="timeout: estouro")])

    result = _run(
        dispatch(
            "explique a prova por indução com detalhes",
            classifier=lambda _prompt: {
                "label": "complexo",
                "model": "phi3.5:latest",
                "reason": "pedido longo",
            },
            client=client,
            fallback_model="phi3.5",
            request_id="req3",
        )
    )

    assert len(client.calls) == 1
    assert result.ok is False
    assert result.redirected is False
    assert result.served_model == "phi3.5:latest"
    assert result.error == "timeout: estouro"


def test_aceita_o_retorno_do_classificador_e_repassa_o_id():
    visto = {}

    def classificar_e_registrar(prompt: str, prompt_id: str | None = None):
        visto["prompt"] = prompt
        visto["prompt_id"] = prompt_id
        return type(
            "Classificacao",
            (),
            {
                "classe": "simples",
                "modelo": "ailo-152m-v2",
                "motivo": "pedido curto (4 palavras <= 25) sem sinais de codigo/raciocinio",
                "palavras": 4,
            },
        )()

    client = FakeClient([OllamaCall(ok=True, latency_s=0.15, text="Paris")])
    result = _run(
        dispatch(
            "Qual é a capital da França?",
            classifier=classificar_e_registrar,
            client=client,
            prompt_id="s01",
            fallback_model="phi3.5",
            request_id="req-clf",
        )
    )

    assert visto == {"prompt": "Qual é a capital da França?", "prompt_id": "s01"}
    assert client.calls == [("ailo-152m-v2", "Qual é a capital da França?")]
    assert result.label == "simples"
    assert result.chosen_model == "ailo-152m-v2"
    assert result.redirected is False


def test_classificador_ausente_nao_chama_o_modelo():
    client = FakeClient([])

    def broken(_prompt: str):
        raise RuntimeError("regra inválida")

    with pytest.raises(Exception, match="regra inválida"):
        _run(
            dispatch(
                "oi",
                classifier=broken,
                client=client,
                fallback_model="phi3.5",
            )
        )

    assert client.calls == []


def test_cliente_mede_latencia_e_devolve_o_texto():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert request.url.path == "/api/generate"
        assert body == {"model": "phi3.5", "prompt": "oi", "stream": False}
        return httpx.Response(200, json={"response": "olá"})

    async def scenario():
        client = OllamaClient(
            "http://ollama.test",
            timeout_s=3,
            transport=httpx.MockTransport(handler),
        )
        try:
            return await client.generate("phi3.5", "oi")
        finally:
            await client.aclose()

    result = _run(scenario())
    assert result.ok is True
    assert result.text == "olá"
    assert result.latency_s >= 0
    assert result.status_code == 200


def test_cliente_trata_timeout_como_falha():
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("estouro")

    async def scenario():
        client = OllamaClient(
            "http://ollama.test",
            timeout_s=1,
            transport=httpx.MockTransport(handler),
        )
        try:
            return await client.generate("phi3.5", "oi")
        finally:
            await client.aclose()

    result = _run(scenario())
    assert result.ok is False
    assert result.error is not None
    assert result.error.startswith("timeout:")
    assert result.latency_s >= 0


def test_cliente_trata_http_de_erro_como_falha():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="model 'foo' not found")

    async def scenario():
        client = OllamaClient(
            "http://ollama.test",
            timeout_s=1,
            transport=httpx.MockTransport(handler),
        )
        try:
            return await client.generate("foo", "oi")
        finally:
            await client.aclose()

    result = _run(scenario())
    assert result.ok is False
    assert result.status_code == 404
    assert "http 404" in (result.error or "")


def test_api_infer_encaminha_e_devolve_o_contrato():
    client = FakeClient([OllamaCall(ok=True, latency_s=0.3, text="42")])
    app = create_app(client=client, classifier=_classify_code)

    with TestClient(app) as http:
        response = http.post(
            "/infer",
            json={"prompt": "escreva uma função", "prompt_id": "c07"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["prompt_id"] == "c07"
    assert body["label"] == "codigo"
    assert body["chosen_model"] == "qwen2.5-coder:0.5b"
    assert body["served_model"] == "qwen2.5-coder:0.5b"
    assert body["redirected"] is False
    assert body["response"] == "42"
    assert "e2e_latency_s" in body


def test_api_devolve_502_quando_o_fallback_tambem_falha():
    client = FakeClient(
        [
            OllamaCall(ok=False, latency_s=0.1, error="conexao: recusada"),
            OllamaCall(ok=False, latency_s=0.1, error="conexao: recusada"),
        ]
    )
    app = create_app(
        client=client,
        classifier=lambda _prompt: Classification("simples", "AILO-152M-v2", "curto"),
    )

    with TestClient(app) as http:
        response = http.post("/infer", json={"prompt": "qual a capital"})

    assert response.status_code == 502
    body = response.json()
    assert body["ok"] is False
    assert body["redirected"] is True
    assert body["served_model"] == FALLBACK_MODEL
    assert body["response"] is None


def test_api_chat_usa_a_ultima_mensagem_do_usuario():
    client = FakeClient([OllamaCall(ok=True, latency_s=0.2, text="pronto")])
    app = create_app(client=client, classifier=_classify_code)

    with TestClient(app) as http:
        response = http.post(
            "/v1/chat/completions",
            json={
                "model": "ignorado",
                "messages": [
                    {"role": "user", "content": "primeira"},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "escreva a função"},
                ],
            },
        )

    assert response.status_code == 200
    assert client.calls[0][1] == "escreva a função"


def test_api_503_sem_classificador(monkeypatch):
    def missing():
        raise ClassifierError(
            "Classificador não encontrado. Esperado: função classificar(prompt) em src/classificador.py."
        )

    monkeypatch.setattr("src.loadbalancer.api.resolve_classifier", missing)
    client = FakeClient([])
    app = create_app(client=client, classifier=None)

    with TestClient(app) as http:
        response = http.post("/infer", json={"prompt": "oi"})

    assert response.status_code == 503
    assert "classificar" in response.json()["error"]
    assert client.calls == []
