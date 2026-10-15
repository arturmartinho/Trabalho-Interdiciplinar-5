"""Núcleo do load balancer: API, cliente do Ollama e despacho."""

from src.loadbalancer.api import create_app
from src.loadbalancer.dispatcher import dispatch

__all__ = ["create_app", "dispatch"]
