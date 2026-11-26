"""Amostragem de RAM/GPU em background para os experimentos.

Uma thread daemon lê RSS, memória do sistema e ``nvidia-smi`` a cada
``interval_ms`` (250 ms por padrão). O custo fica fora do caminho da
requisição: a latência medida no runner não inclui essa coleta. Leituras que
falham reaproveitam o último valor válido; se nunca houve leitura válida, o
campo fica vazio e o motivo aparece em ``resource_error``.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from typing import Any, Callable

GPU_AVG_PEAK = ("gpu_util_pct", "gpu_mem_used_mb", "gpu_power_w")

NVIDIA_QUERY = (
    "utilization.gpu,memory.used,memory.total,power.draw"
)


def _numero(valor: str) -> float | None:
    texto = valor.strip().replace("MB", "").replace("W", "").strip()
    if not texto or texto.upper() == "N/A":
        return None
    try:
        return float(texto)
    except ValueError:
        return None


def agregar_gpus(gpus: list[dict[str, float | None]]) -> dict[str, Any]:
    """Agrega todas as GPUs em uma amostra.

    Regra registrada nos metadados: utilização é a média das GPUs da amostra;
    memória usada/total e potência são somas. O pico entre amostras é
    calculado em ``resumo()``.
    """
    if not gpus:
        return {}
    util = [g["util_pct"] for g in gpus if g.get("util_pct") is not None]
    usado = [g["mem_used_mb"] for g in gpus if g.get("mem_used_mb") is not None]
    total = [g["mem_total_mb"] for g in gpus if g.get("mem_total_mb") is not None]
    potencia = [g["power_w"] for g in gpus if g.get("power_w") is not None]

    def media(valores: list[float]) -> float | None:
        return round(sum(valores) / len(valores), 2) if valores else None

    def soma(valores: list[float]) -> float | None:
        return round(sum(valores), 2) if valores else None

    util_media = media(util)
    usado_total = soma(usado)
    potencia_total = soma(potencia)
    return {
        "gpu_available": True,
        "gpu_count": len(gpus),
        "gpu_util_pct": util_media,
        "gpu_mem_used_mb": usado_total,
        "gpu_mem_total_mb": soma(total),
        "gpu_power_w": potencia_total,
    }


def consultar_nvidia_smi(timeout_s: float = 3.0) -> list[dict[str, float | None]]:
    """Consulta todas as GPUs uma vez; ``nvidia-smi`` roda só na thread."""
    completado = subprocess.run(
        ["nvidia-smi", f"--query-gpu={NVIDIA_QUERY}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if completado.returncode != 0:
        detalhe = (completado.stderr or completado.stdout).strip().splitlines()
        raise RuntimeError(detalhe[0] if detalhe else f"nvidia-smi saiu com {completado.returncode}")

    gpus: list[dict[str, float | None]] = []
    for linha in completado.stdout.splitlines():
        colunas = [parte.strip() for parte in linha.split(",")]
        if len(colunas) < 4 or not any(colunas):
            continue
        gpus.append({
            "util_pct": _numero(colunas[0]),
            "mem_used_mb": _numero(colunas[1]),
            "mem_total_mb": _numero(colunas[2]),
            "power_w": _numero(colunas[3]),
        })
    return gpus


class ResourceSampler:
    """Coleta amostras em background e resume média/pico por bloco."""

    def __init__(
        self,
        interval_ms: int = 250,
        gpu_metrics: str = "auto",
        *,
        psutil_module: Any = None,
        gpu_query: Callable[[], list[dict[str, float | None]]] | None = None,
    ) -> None:
        if interval_ms < 1:
            raise ValueError("interval_ms precisa ser >= 1")
        if gpu_metrics not in ("auto", "off", "nvidia-smi"):
            raise ValueError("gpu_metrics deve ser auto, off ou nvidia-smi")
        self.interval_ms = int(interval_ms)
        self.gpu_metrics = gpu_metrics
        self._psutil = psutil_module
        self._gpu_query = gpu_query
        self._processo: Any = None
        self._amostras: list[dict[str, Any]] = []
        self._cache_ram: dict[str, float] = {}
        self._cache_gpu: dict[str, Any] = {}
        self._erros: set[str] = set()
        self._trava = threading.Lock()
        self._parar = threading.Event()
        self._thread: threading.Thread | None = None
        self.gpu_available = False
        self.gpu_count = 0

    # ------------------------------------------------------------------ #
    # Coleta
    # ------------------------------------------------------------------ #
    def _carregar_psutil(self) -> Any:
        if self._psutil is None:
            try:
                import psutil
            except ImportError:
                self._erros.add("psutil indisponível (instale o extra experiments)")
                return None
            self._psutil = psutil
        return self._psutil

    def _query_gpu(self) -> Callable[[], list[dict[str, float | None]]]:
        if self._gpu_query is not None:
            return self._gpu_query
        if self.gpu_metrics == "auto" and not shutil.which("nvidia-smi"):
            return lambda: []
        return consultar_nvidia_smi

    def coletar_amostra(self) -> dict[str, Any]:
        """Lê uma amostra, usando cache quando a leitura falha."""
        amostra: dict[str, Any] = {}

        psutil = self._carregar_psutil()
        if psutil is not None:
            try:
                if self._processo is None:
                    self._processo = psutil.Process()
                rss = self._processo.memory_info().rss / (1024 * 1024)
                memoria = psutil.virtual_memory()
                self._cache_ram = {
                    "ram_rss_mb": round(rss, 2),
                    "ram_system_used_mb": round(memoria.used / (1024 * 1024), 2),
                    "ram_system_pct": round(memoria.percent, 2),
                }
            except Exception as exc:  # medição nunca derruba a rodada
                self._erros.add(f"psutil: {type(exc).__name__}: {exc}")
        amostra.update(self._cache_ram)

        if self.gpu_metrics != "off":
            try:
                gpus = self._query_gpu()()
                if gpus:
                    self._cache_gpu = agregar_gpus(gpus)
                    self.gpu_available = True
                    self.gpu_count = len(gpus)
                else:
                    self.gpu_available = False
                    self.gpu_count = 0
            except Exception as exc:
                self._erros.add(f"gpu: {type(exc).__name__}: {exc}")
        amostra.update(self._cache_gpu)

        with self._trava:
            self._amostras.append(amostra)
        return amostra

    def _loop(self) -> None:
        while not self._parar.is_set():
            self.coletar_amostra()
            if self._parar.wait(self.interval_ms / 1000):
                break

    # ------------------------------------------------------------------ #
    # Ciclo de vida e resumo
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._parar.clear()
        self._thread = threading.Thread(
            target=self._loop, name="resource-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._parar.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 4 * self.interval_ms / 1000))
            self._thread = None
        with self._trava:
            vazio = not self._amostras
        if vazio:  # bloco curto demais para a thread: uma leitura direta
            self.coletar_amostra()
        return self.resumo()

    def reset(self) -> None:
        """Descarta amostras/erros do bloco anterior (mantém cache e disponibilidade)."""
        with self._trava:
            self._amostras = []
            self._erros = set()

    def __enter__(self) -> "ResourceSampler":
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()

    def resumo(self) -> dict[str, Any]:
        with self._trava:
            amostras = list(self._amostras)

        resultado: dict[str, Any] = {}
        for chave in ("ram_rss_mb", "ram_system_used_mb", "ram_system_pct"):
            avg, peak = _avg_peak(amostras, chave)
            resultado[f"{chave}_avg"] = avg
            resultado[f"{chave}_peak"] = peak

        for chave in GPU_AVG_PEAK:
            avg, peak = _avg_peak(amostras, chave)
            resultado[f"{chave}_avg"] = avg
            resultado[f"{chave}_peak"] = peak
        resultado["gpu_mem_total_mb"] = _ultimo(amostras, "gpu_mem_total_mb")
        resultado["gpu_available"] = self.gpu_available
        resultado["gpu_count"] = self.gpu_count if self.gpu_available else 0
        resultado["gpu_sample_count"] = sum(
            1 for amostra in amostras if amostra.get("gpu_util_pct") is not None
        )
        resultado["gpu_sample_interval_ms"] = self.interval_ms
        resultado["resource_error"] = "; ".join(sorted(self._erros)) if self._erros else ""
        return resultado


def _avg_peak(amostras: list[dict[str, Any]], chave: str) -> tuple[float | None, float | None]:
    valores = [float(a[chave]) for a in amostras if a.get(chave) is not None]
    if not valores:
        return None, None
    return round(sum(valores) / len(valores), 2), round(max(valores), 2)


def _ultimo(amostras: list[dict[str, Any]], chave: str) -> float | None:
    for amostra in reversed(amostras):
        if amostra.get(chave) is not None:
            return float(amostra[chave])
    return None
