"""Bateria de prompts reais (PT/EN) contra o load balancer ao vivo."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

PROMPTS = [
    # --- PT simples ---
    {"id": "pt-s1", "lang": "pt", "esperado": "simples", "prompt": "Qual e a capital do Brasil?"},
    {"id": "pt-s2", "lang": "pt", "esperado": "simples", "prompt": "Quantos dias tem uma semana?"},
    {"id": "pt-s3", "lang": "pt", "esperado": "simples", "prompt": "Qual e o resultado de 12 vezes 8?"},
    # --- EN simples ---
    {"id": "en-s1", "lang": "en", "esperado": "simples", "prompt": "What is the capital of Canada?"},
    {"id": "en-s2", "lang": "en", "esperado": "simples", "prompt": "How many hours are in a day?"},
    {"id": "en-s3", "lang": "en", "esperado": "simples", "prompt": "What color is a ripe banana?"},
    # --- PT codigo ---
    {
        "id": "pt-c1",
        "lang": "pt",
        "esperado": "codigo",
        "prompt": "Escreva uma funcao em Python que receba uma lista e devolva apenas os numeros pares.",
    },
    {
        "id": "pt-c2",
        "lang": "pt",
        "esperado": "codigo",
        "prompt": "Monte uma consulta SQL que liste o nome e o email dos usuarios ativos.",
    },
    {
        "id": "pt-c3",
        "lang": "pt",
        "esperado": "codigo",
        "prompt": "Corrija este bug: def media(nums): return sum(nums) / 0",
    },
    # --- EN codigo ---
    {
        "id": "en-c1",
        "lang": "en",
        "esperado": "codigo",
        "prompt": "Write a Python function that counts how many vowels are in a string.",
    },
    {
        "id": "en-c2",
        "lang": "en",
        "esperado": "codigo",
        "prompt": "Write a JavaScript function that reverses an array without using reverse().",
    },
    {
        "id": "en-c3",
        "lang": "en",
        "esperado": "codigo",
        "prompt": "Fix this bug: def foo(x): return x / 0",
    },
    # --- PT complexo ---
    {
        "id": "pt-x1",
        "lang": "pt",
        "esperado": "complexo",
        "prompt": "Explique passo a passo por que o ceu parece azul durante o dia.",
    },
    {
        "id": "pt-x2",
        "lang": "pt",
        "esperado": "complexo",
        "prompt": "Compare vantagens e desvantagens de trabalhar com microservicos versus um monolitico.",
    },
    {
        "id": "pt-x3",
        "lang": "pt",
        "esperado": "complexo",
        "prompt": "Analise e justifique como a inflacao alta afeta o poder de compra das familias.",
    },
    # --- EN complexo ---
    {
        "id": "en-x1",
        "lang": "en",
        "esperado": "complexo",
        "prompt": "Explain step by step why ice floats on water.",
    },
    {
        "id": "en-x2",
        "lang": "en",
        "esperado": "complexo",
        "prompt": "Compare the pros and cons of remote work versus office work.",
    },
    {
        "id": "en-x3",
        "lang": "en",
        "esperado": "complexo",
        "prompt": "Analyze why renewable energy adoption is growing worldwide.",
    },
    # --- falsos positivos / ambiguos ---
    {
        "id": "pt-fp1",
        "lang": "pt",
        "esperado": "simples",
        "prompt": "Qual e o programa de TV mais famoso do Brasil?",
    },
    {
        "id": "pt-fp2",
        "lang": "pt",
        "esperado": "simples",
        "prompt": "A api do banco central publica a taxa Selic todo dia?",
    },
    {
        "id": "en-fp1",
        "lang": "en",
        "esperado": "simples",
        "prompt": "What is the best TV program for kids on Saturday morning?",
    },
    {
        "id": "pt-a1",
        "lang": "pt",
        "esperado": "codigo",
        "prompt": "Explique esta funcao em Python e implemente testes unitarios.",
    },
]

URL = "http://127.0.0.1:8000/infer"
OUT = Path(__file__).with_name("bench_prompts_resultado.json")


def main() -> None:
    rows: list[dict] = []
    with httpx.Client(timeout=180.0) as client:
        for item in PROMPTS:
            t0 = time.perf_counter()
            try:
                resp = client.post(
                    URL,
                    json={"prompt": item["prompt"], "prompt_id": item["id"]},
                )
                data = resp.json()
                wall = round(time.perf_counter() - t0, 3)
                text = data.get("response") or ""
                preview = text.replace("\n", " ")[:180]
                route_ok = data.get("label") == item["esperado"]
                rows.append(
                    {
                        "id": item["id"],
                        "lang": item["lang"],
                        "esperado": item["esperado"],
                        "label": data.get("label"),
                        "route_ok": route_ok,
                        "chosen": data.get("chosen_model"),
                        "served": data.get("served_model"),
                        "redirected": data.get("redirected"),
                        "ok": data.get("ok"),
                        "http": resp.status_code,
                        "lat_e2e": data.get("e2e_latency_s"),
                        "lat_wall": wall,
                        "resp_chars": len(text),
                        "preview": preview,
                        "prompt": item["prompt"],
                    }
                )
                print(
                    f"OK {item['id']:8} route={str(data.get('label')):8} "
                    f"esp={item['esperado']:8} match={route_ok} "
                    f"lat={data.get('e2e_latency_s')}"
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "id": item["id"],
                        "error": str(exc),
                        "route_ok": False,
                        "ok": False,
                        "prompt": item["prompt"],
                    }
                )
                print(f"ERR {item['id']}: {exc}")

    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    n = len(rows)
    route_hits = sum(1 for r in rows if r.get("route_ok"))
    ok_hits = sum(1 for r in rows if r.get("ok"))
    lats = [r["lat_e2e"] for r in rows if isinstance(r.get("lat_e2e"), (int, float))]
    print("---SUMMARY---")
    print(f"total={n} route_ok={route_hits}/{n} infer_ok={ok_hits}/{n}")
    if lats:
        print(
            f"lat_avg={sum(lats)/len(lats):.3f}s "
            f"lat_min={min(lats):.3f}s lat_max={max(lats):.3f}s"
        )
    print(f"saved={OUT}")


if __name__ == "__main__":
    main()
