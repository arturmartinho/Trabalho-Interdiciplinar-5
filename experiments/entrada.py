"""Carregamento e validação do conjunto de prompts de experimento."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CLASSES = ("simples", "codigo", "complexo")
CAMPOS_TEXTO = ("idioma", "subtipo", "linguagem", "dificuldade", "gerado_por")


def carregar_prompts(caminho: str | Path) -> list[dict[str, Any]]:
    """Lê uma lista de prompts e falha cedo com erro claro.

    Contrato mínimo: ``id`` e ``prompt`` não vazios e ``classe_esperada`` em
    {simples, codigo, complexo}. Os metadados de grupo são preservados; os
    campos de texto são normalizados para ``str`` (vazio quando ausentes).
    """
    origem = Path(caminho)
    try:
        bruto = json.loads(origem.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"arquivo de prompts não encontrado: {origem}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON inválido em {origem}: {exc}") from exc
    if not isinstance(bruto, list) or not bruto:
        raise ValueError(f"{origem}: esperada uma lista não vazia de prompts")

    prompts: list[dict[str, Any]] = []
    vistos: set[str] = set()
    for posicao, item in enumerate(bruto, 1):
        if not isinstance(item, dict):
            raise ValueError(f"prompt #{posicao}: esperado um objeto JSON")
        pid = str(item.get("id") or "").strip()
        prompt = str(item.get("prompt") or "").strip()
        classe = str(item.get("classe_esperada") or "").strip().lower()
        if not pid:
            raise ValueError(f"prompt #{posicao}: `id` é obrigatório")
        if pid in vistos:
            raise ValueError(f"prompt #{posicao}: `id` duplicado ({pid})")
        if not prompt:
            raise ValueError(f"prompt #{posicao} ({pid}): `prompt` é obrigatório")
        if classe not in CLASSES:
            raise ValueError(
                f"prompt #{posicao} ({pid}): `classe_esperada` deve ser um de "
                f"{', '.join(CLASSES)} (recebido: {classe or 'vazio'})"
            )
        vistos.add(pid)
        normalizado = dict(item)
        normalizado.update({
            "id": pid,
            "prompt": prompt,
            "classe_esperada": classe,
        })
        for campo in CAMPOS_TEXTO:
            normalizado[campo] = str(item.get(campo) or "").strip()
        prompts.append(normalizado)
    return prompts
