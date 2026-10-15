"""Sobe a API de verdade e fala com um Ollama compatível na máquina.

O classificador usado é o da branch feat/classificador. O teste grava
src/classificador.py só enquanto roda e apaga em seguida, se o arquivo
não existia antes.
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
CLASSIFICADOR = ROOT / "src" / "classificador.py"

PROMPTS = {
    "simples": "Qual é a capital da França?",
    "codigo": "Escreva uma função em Python que inverte uma string.",
    "complexo": "Explique passo a passo por que o céu é azul.",
}
MODELOS = {
    "simples": "ailo-152m-v2",
    "codigo": "qwen2.5-coder:0.5b",
    "complexo": "phi3.5",
}


class _OllamaHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] != "/api/tags":
            self._send(404, {"error": "rota inexistente"})
            return
        nomes = [{"name": nome} for nome in MODELOS.values()]
        self._send(200, {"models": nomes})

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/api/generate":
            self._send(404, {"error": "rota inexistente"})
            return
        tamanho = int(self.headers.get("Content-Length", "0"))
        bruto = self.rfile.read(tamanho)
        pedido = json.loads(bruto.decode())
        modelo = str(pedido.get("model", ""))
        self.server.chamadas.append(modelo)  # type: ignore[attr-defined]
        if modelo not in MODELOS.values() or modelo in self.server.falhos:  # type: ignore[attr-defined]
            self._send(404, {"error": f"model '{modelo}' not found"})
            return
        self._send(200, {"model": modelo, "response": f"ok:{modelo}", "done": True})

    def _send(self, status: int, payload: dict) -> None:
        corpo = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(corpo)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(corpo)

    def log_message(self, fmt: str, *args) -> None:
        return


@pytest.fixture(scope="module")
def classificador_disponivel():
    """Garante o módulo do Aluno 2 no lugar em que o despacho importa."""
    ja_existia = CLASSIFICADOR.exists()
    if not ja_existia:
        codigo = subprocess.check_output(
            ["git", "show", "origin/feat/classificador:src/classificador.py"],
            cwd=ROOT,
        )
        CLASSIFICADOR.write_bytes(codigo)
    yield
    if not ja_existia and CLASSIFICADOR.exists():
        CLASSIFICADOR.unlink()
        cache = ROOT / "src" / "__pycache__"
        for arquivo in cache.glob("classificador*.pyc"):
            arquivo.unlink()


@pytest.fixture
def ollama_falso():
    servidor = ThreadingHTTPServer(("127.0.0.1", 0), _OllamaHandler)
    servidor.chamadas = []
    servidor.falhos = set()
    thread = threading.Thread(target=servidor.serve_forever, daemon=True)
    thread.start()
    try:
        yield servidor
    finally:
        servidor.shutdown()
        servidor.server_close()
        thread.join(timeout=5)


def _porta_livre() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _esperar(url: str, timeout_s: float = 20) -> httpx.Response:
    limite = time.time() + timeout_s
    ultimo_erro: Exception | None = None
    while time.time() < limite:
        try:
            return httpx.get(url, timeout=1.0)
        except httpx.HTTPError as exc:
            ultimo_erro = exc
            time.sleep(0.15)
    raise AssertionError(f"a API não subiu em {url}: {ultimo_erro}")


def _subir_api(ollama_falso, porta: int, extra_env: dict | None = None):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["OLLAMA_HOST"] = f"http://127.0.0.1:{ollama_falso.server_address[1]}"
    env["OLLAMA_TIMEOUT_SECONDS"] = "5"
    env["LOADBALANCER_HOST"] = "127.0.0.1"
    env["LOADBALANCER_PORT"] = str(porta)
    env["FALLBACK_MODEL"] = "phi3.5"
    env["MODELO_SIMPLES"] = MODELOS["simples"]
    env["MODELO_CODIGO"] = MODELOS["codigo"]
    env["MODELO_COMPLEXO"] = MODELOS["complexo"]
    if extra_env:
        env.update(extra_env)
    log = Path(os.environ.get("TEMP", ".")) / f"loadbalancer-{porta}.log"
    handle = log.open("w", encoding="utf-8")
    processo = subprocess.Popen(
        [sys.executable, "-m", "src.loadbalancer"],
        cwd=ROOT,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    return processo, log, handle


def _derrubar(processo: subprocess.Popen, handle) -> None:
    if processo.poll() is None:
        processo.terminate()
        try:
            processo.wait(timeout=5)
        except subprocess.TimeoutExpired:
            processo.kill()
            processo.wait(timeout=5)
    handle.close()


def test_processo_real_roteia_os_tres_tipos(classificador_disponivel, ollama_falso):
    porta = _porta_livre()
    processo, log, handle = _subir_api(ollama_falso, porta)
    base = f"http://127.0.0.1:{porta}"
    try:
        saude = _esperar(f"{base}/health")
        assert saude.status_code == 200
        corpo_saude = saude.json()
        assert corpo_saude["ok"] is True
        assert set(corpo_saude["models"]) == set(MODELOS.values())

        docs = httpx.get(f"{base}/docs", timeout=5)
        assert docs.status_code == 200

        vazio = httpx.post(f"{base}/infer", json={"prompt": ""}, timeout=5)
        assert vazio.status_code == 422

        for classe, prompt in PROMPTS.items():
            resposta = httpx.post(
                f"{base}/infer",
                json={"prompt": prompt, "prompt_id": classe},
                timeout=10,
            )
            assert resposta.status_code == 200, resposta.text
            corpo = resposta.json()
            assert corpo["ok"] is True
            assert corpo["label"] == classe
            assert corpo["prompt_id"] == classe
            assert corpo["chosen_model"] == MODELOS[classe]
            assert corpo["served_model"] == MODELOS[classe]
            assert corpo["redirected"] is False
            assert corpo["response"] == f"ok:{MODELOS[classe]}"
            assert corpo["e2e_latency_s"] >= 0
            assert corpo["ollama_latency_s"] >= 0

        conversa = httpx.post(
            f"{base}/v1/chat/completions",
            json={
                "model": "ignorado",
                "messages": [
                    {"role": "user", "content": "primeira"},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": PROMPTS["codigo"]},
                ],
            },
            timeout=10,
        )
        assert conversa.status_code == 200, conversa.text
        assert conversa.json()["chosen_model"] == MODELOS["codigo"]
        assert conversa.json()["served_model"] == MODELOS["codigo"]
    finally:
        _derrubar(processo, handle)

    assert processo.returncode is not None
    texto = log.read_text(encoding="utf-8")
    assert "[classificador]" in texto
    assert "despacho" in texto
    assert "classe=simples" in texto
    assert "classe=codigo" in texto
    assert "classe=complexo" in texto
    assert ollama_falso.chamadas == [
        MODELOS["simples"],
        MODELOS["codigo"],
        MODELOS["complexo"],
        MODELOS["codigo"],
    ]


def test_processo_real_redireciona_quando_o_modelo_nao_existe(classificador_disponivel, ollama_falso):
    porta = _porta_livre()
    processo, log, handle = _subir_api(
        ollama_falso,
        porta,
        extra_env={"MODELO_CODIGO": "modelo-inexistente"},
    )
    base = f"http://127.0.0.1:{porta}"
    try:
        _esperar(f"{base}/health")
        resposta = httpx.post(
            f"{base}/infer",
            json={"prompt": PROMPTS["codigo"], "prompt_id": "fb"},
            timeout=10,
        )
        assert resposta.status_code == 200, resposta.text
        corpo = resposta.json()
        assert corpo["ok"] is True
        assert corpo["label"] == "codigo"
        assert corpo["chosen_model"] == "modelo-inexistente"
        assert corpo["redirected"] is True
        assert corpo["served_model"] == "phi3.5"
        assert corpo["response"] == "ok:phi3.5"
        assert "not found" in (corpo["redirect_reason"] or "")
    finally:
        _derrubar(processo, handle)

    texto = log.read_text(encoding="utf-8")
    assert "redirecionamento" in texto
    assert "modelo_escolhido=modelo-inexistente" in texto
    assert "fallback=phi3.5" in texto
    assert ollama_falso.chamadas == ["modelo-inexistente", "phi3.5"]


def test_processo_real_devolve_502_se_o_fallback_tambem_falha(classificador_disponivel, ollama_falso):
    ollama_falso.falhos.add("phi3.5")
    ollama_falso.falhos.add("modelo-inexistente")
    porta = _porta_livre()
    processo, log, handle = _subir_api(
        ollama_falso,
        porta,
        extra_env={"MODELO_CODIGO": "modelo-inexistente"},
    )
    base = f"http://127.0.0.1:{porta}"
    try:
        _esperar(f"{base}/health")
        resposta = httpx.post(
            f"{base}/infer",
            json={"prompt": PROMPTS["codigo"], "prompt_id": "fb-falha"},
            timeout=10,
        )
        assert resposta.status_code == 502, resposta.text
        corpo = resposta.json()
        assert corpo["ok"] is False
        assert corpo["redirected"] is True
        assert corpo["served_model"] == "phi3.5"
        assert corpo["response"] is None
    finally:
        _derrubar(processo, handle)

    assert "redirecionamento" in log.read_text(encoding="utf-8")
