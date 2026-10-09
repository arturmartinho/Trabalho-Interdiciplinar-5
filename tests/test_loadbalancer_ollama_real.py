"""Exercita a API contra o Ollama instalado na máquina.

Pula se o serviço local não estiver no ar. A geração fica curta de propósito,
via OLLAMA_NUM_PREDICT, para o teste terminar mesmo com o Phi na CPU.
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
CLASSIFICADOR = ROOT / "src" / "classificador.py"
OLLAMA = "http://127.0.0.1:11434"

PROMPTS = {
    "simples": ("Qual é a capital da França?", "Alieno/ailo-152m-v2"),
    "codigo": ("Escreva uma função em Python que inverte uma string.", "qwen2.5-coder:0.5b"),
    "complexo": ("Explique passo a passo por que o céu é azul.", "phi3.5"),
}


def _ollama_no_ar() -> bool:
    try:
        resposta = httpx.get(f"{OLLAMA}/api/tags", timeout=5)
    except httpx.HTTPError:
        return False
    if resposta.status_code != 200:
        return False
    nomes = " ".join(
        item.get("name", "")
        for item in resposta.json().get("models", [])
        if isinstance(item, dict)
    )
    return all(tag in nomes for tag in ("ailo-152m-v2", "qwen2.5-coder:0.5b", "phi3.5"))


pytestmark = pytest.mark.skipif(not _ollama_no_ar(), reason="Ollama local sem os três modelos")


@pytest.fixture(scope="module")
def classificador_disponivel():
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
        for arquivo in (ROOT / "src" / "__pycache__").glob("classificador*.pyc"):
            arquivo.unlink()


def _porta_livre() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _esperar(url: str, timeout_s: float = 20) -> httpx.Response:
    limite = time.time() + timeout_s
    ultimo: Exception | None = None
    while time.time() < limite:
        try:
            return httpx.get(url, timeout=2)
        except httpx.HTTPError as exc:
            ultimo = exc
            time.sleep(0.2)
    raise AssertionError(f"a API não subiu: {ultimo}")


def _subir(porta: int, extra: dict | None = None):
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "OLLAMA_HOST": OLLAMA,
            "OLLAMA_TIMEOUT_SECONDS": "180",
            "OLLAMA_NUM_PREDICT": "32",
            "LOADBALANCER_HOST": "127.0.0.1",
            "LOADBALANCER_PORT": str(porta),
            "MODELO_SIMPLES": "Alieno/ailo-152m-v2",
            "MODELO_CODIGO": "qwen2.5-coder:0.5b",
            "MODELO_COMPLEXO": "phi3.5",
            "FALLBACK_MODEL": "phi3.5",
        }
    )
    if extra:
        env.update(extra)
    log = Path(os.environ.get("TEMP", ".")) / f"loadbalancer-real-{porta}.log"
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
            processo.wait(timeout=8)
        except subprocess.TimeoutExpired:
            processo.kill()
            processo.wait(timeout=8)
    handle.close()


def test_ollama_real_roteia_os_tres_modelos(classificador_disponivel):
    porta = _porta_livre()
    processo, log, handle = _subir(porta)
    base = f"http://127.0.0.1:{porta}"
    try:
        saude = _esperar(f"{base}/health")
        assert saude.status_code == 200, saude.text
        instalados = " ".join(saude.json()["models"])
        assert "ailo-152m-v2" in instalados
        assert "qwen2.5-coder:0.5b" in instalados
        assert "phi3.5" in instalados

        for classe, (prompt, modelo) in PROMPTS.items():
            resposta = httpx.post(
                f"{base}/infer",
                json={"prompt": prompt, "prompt_id": classe},
                timeout=180,
            )
            assert resposta.status_code == 200, resposta.text
            corpo = resposta.json()
            assert corpo["ok"] is True
            assert corpo["label"] == classe
            assert corpo["chosen_model"] == modelo
            assert corpo["served_model"] == modelo
            assert corpo["redirected"] is False
            assert corpo["response"]
            assert corpo["e2e_latency_s"] > 0
    finally:
        _derrubar(processo, handle)

    texto = log.read_text(encoding="utf-8")
    assert "[classificador]" in texto
    assert "classe=simples" in texto
    assert "classe=codigo" in texto
    assert "classe=complexo" in texto


def test_ollama_real_redireciona_para_o_phi(classificador_disponivel):
    porta = _porta_livre()
    processo, log, handle = _subir(porta, {"MODELO_CODIGO": "modelo-inexistente"})
    base = f"http://127.0.0.1:{porta}"
    try:
        _esperar(f"{base}/health")
        resposta = httpx.post(
            f"{base}/infer",
            json={
                "prompt": "Escreva uma função em Python que inverte uma string.",
                "prompt_id": "fallback",
            },
            timeout=180,
        )
        assert resposta.status_code == 200, resposta.text
        corpo = resposta.json()
        assert corpo["label"] == "codigo"
        assert corpo["chosen_model"] == "modelo-inexistente"
        assert corpo["redirected"] is True
        assert corpo["served_model"] == "phi3.5"
        assert corpo["response"]
    finally:
        _derrubar(processo, handle)

    texto = log.read_text(encoding="utf-8")
    assert "redirecionamento" in texto
    assert "fallback=phi3.5" in texto
