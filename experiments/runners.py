"""Runners dos cenários de experimento.

``default_model`` chama o Ollama diretamente com o mesmo tag do baseline;
``loadbalancer_classifier`` chama ``POST /infer``. Ambos devolvem um dicionário
com as colunas dinâmicas da medição; o runner de experimento só mede o tempo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.loadbalancer.usage import extract_usage, sem_usage

from experiments.analise_metricas import CENARIO_LB, CENARIO_PADRAO
from experiments.config_experimento import (
    LB_LOCAL_URL,
    NUM_PREDICT,
    OLLAMA_HOST,
    modelo_padrao,
)

ALIASES = {"baseline": CENARIO_PADRAO, "proposta": CENARIO_LB}


@dataclass(frozen=True)
class ConfigExecucao:
    """Endpoints efetivos por deployment e tag comum aos dois cenários."""

    model_tag: str = ""
    ollama_local: str = OLLAMA_HOST
    ollama_remote: str = ""
    lb_local_url: str = LB_LOCAL_URL
    lb_remote_url: str = ""
    num_predict: str = NUM_PREDICT

    def __post_init__(self) -> None:
        if not self.model_tag.strip():
            object.__setattr__(self, "model_tag", modelo_padrao())

    def ollama_host(self, deployment: str) -> str:
        host = self.ollama_remote if deployment == "remote" else self.ollama_local
        if not host:
            raise ValueError(f"deployment '{deployment}' sem host Ollama configurado")
        return host.rstrip("/")

    def lb_url(self, deployment: str) -> str:
        url = self.lb_remote_url if deployment == "remote" else self.lb_local_url
        if not url:
            raise ValueError(f"deployment '{deployment}' sem URL do load balancer configurada")
        return url.rstrip("/")

    def effective(self, deployment: str, scenario: str) -> dict[str, str]:
        """Valores efetivamente usados, registrados no CSV e nos metadados."""
        return {
            "effective_ollama_host": self.ollama_host(deployment),
            "effective_model_tag": self.model_tag,
            "effective_lb_url": self.lb_url(deployment) if scenario == CENARIO_LB else "",
        }


def normalizar_cenario(cenario: str) -> str:
    valor = str(cenario).strip().lower()
    return ALIASES.get(valor, valor)


def chamar_baseline(
    client: Any,
    prompt: str,
    *,
    model_tag: str | None = None,
    ollama_host: str | None = None,
    num_predict: str | None = None,
) -> dict[str, Any]:
    """Chamada direta ao Ollama; corpo idêntico ao do ``OllamaClient``."""
    tag = model_tag or modelo_padrao()
    host = (ollama_host or OLLAMA_HOST).rstrip("/")
    corpo: dict[str, Any] = {"model": tag, "prompt": prompt, "stream": False}
    limite = num_predict if num_predict is not None else NUM_PREDICT
    if limite:
        corpo["options"] = {"num_predict": int(limite)}

    resposta = client.post(f"{host}/api/generate", json=corpo)
    resposta.raise_for_status()
    payload = resposta.json()
    texto = str(payload.get("response") or "").strip()
    if not texto:
        raise ValueError("resposta vazia")  # o balancer também trata vazio como falha

    return {
        "classe": "",
        "modelo_escolhido": tag,
        "modelo_atendeu": tag,
        "desvio": False,
        "lb_e2e_s": "",
        "ollama_latency_s": "",
        "resposta_chars": len(texto),
        **extract_usage(payload),
    }


def chamar_proposta(
    client: Any,
    prompt: str,
    prompt_id: str,
    *,
    lb_url: str | None = None,
) -> dict[str, Any]:
    """Chamada ao LB; em falha o balancer devolve 502 com JSON do despacho."""
    url = (lb_url or LB_LOCAL_URL).rstrip("/")
    resposta = client.post(f"{url}/infer", json={"prompt": prompt, "prompt_id": prompt_id})
    dados = resposta.json()
    if resposta.status_code >= 400 or not dados.get("ok", False):
        raise RuntimeError(f"http {resposta.status_code}: {dados.get('error')}")

    usage = {chave: dados[chave] for chave in (
        "prompt_tokens", "completion_tokens", "total_tokens",
        "prompt_eval_duration_s", "eval_duration_s", "load_duration_s", "total_duration_s",
        "token_source",
    ) if dados.get(chave) is not None}
    if "prompt_tokens" in usage or "completion_tokens" in usage:
        usage.setdefault("token_source", "ollama")
    else:
        usage = sem_usage("lb_unavailable")

    predita = dados.get("label", "") or ""
    return {
        "classe": predita,
        "modelo_escolhido": dados.get("chosen_model", ""),
        "modelo_atendeu": dados.get("served_model", ""),
        "desvio": dados.get("redirected", ""),
        "lb_e2e_s": dados.get("e2e_latency_s", ""),
        "ollama_latency_s": dados.get("ollama_latency_s", ""),
        "resposta_chars": len(dados.get("response") or ""),
        **usage,
    }


class DefaultModelRunner:
    """Cenário sem classificador: mesmo tag nos dois deployments."""

    def call(self, client: Any, item: dict[str, Any], cfg: ConfigExecucao, deployment: str) -> dict[str, Any]:
        return chamar_baseline(
            client,
            item["prompt"],
            model_tag=cfg.model_tag,
            ollama_host=cfg.ollama_host(deployment),
            num_predict=cfg.num_predict,
        )


class LoadBalancerRunner:
    """Cenário com classificador + despacho + fallback existente."""

    def call(self, client: Any, item: dict[str, Any], cfg: ConfigExecucao, deployment: str) -> dict[str, Any]:
        return chamar_proposta(
            client,
            item["prompt"],
            item["id"],
            lb_url=cfg.lb_url(deployment),
        )


def runner_para(cenario: str, cfg: ConfigExecucao, deployment: str) -> Any:
    nome = normalizar_cenario(cenario)
    if nome == CENARIO_PADRAO:
        return DefaultModelRunner()
    if nome == CENARIO_LB:
        return LoadBalancerRunner()
    raise ValueError(f"cenário desconhecido: {cenario}")


def classificacao(cenario: str, predita: str, esperada: str) -> dict[str, Any]:
    """``classification_error`` é 0/1; vazio quando não há classificador."""
    if normalizar_cenario(cenario) != CENARIO_LB or not predita or not esperada:
        return {"classification_correct": "", "classification_error": ""}
    correto = int(predita == esperada)
    return {"classification_correct": correto, "classification_error": 1 - correto}
