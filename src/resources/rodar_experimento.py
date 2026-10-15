"""Roda baseline e proposta sobre o mesmo conjunto de prompts e grava resultados.csv.

Baseline: todo prompt vai direto ao Phi-3.5-mini no Ollama (sem classificador).
Proposta: todo prompt vai ao POST /infer do load balancer (vale a classificação).

Medição: latência de ponta a ponta no cliente, com time.perf_counter(), nos dois cenários. O corpo enviado
ao Ollama no baseline é idêntico ao que o balancer envia (model, prompt, stream=false e, se existir,
options.num_predict vindo de OLLAMA_NUM_PREDICT). Sem limite igual, a latência mediria quanto cada
modelo decidiu escrever, não a velocidade de cada um.

Delineamento de cada rodada (contrabalançado, para não favorecer nenhum cenário):
  - o mesmo conjunto de prompts passa pelos dois cenários, uma vez cada por rodada;
  - passo 1: todos os prompts em ordem embaralhada, cada um no cenário que lhe coube ir primeiro;
  - passo 2: os mesmos prompts, em outra ordem, no outro cenário;
  - quem vai primeiro é balanceado dentro de cada classe (metade baseline, metade proposta);
  - os dois pedidos do mesmo prompt ficam separados por >= n/2 chamadas, e o mesmo prompt nunca roda duas
    vezes seguidas. Assim o Ollama não reaproveita o cache do prompt anterior (nos complexos os dois
    cenários usam o Phi, e rodar um logo após o outro favoreceria quem vai depois).

Uso:
  python rodar_experimento.py --repeticoes 3 --aquecimento 2
Pré-requisitos: Ollama no ar com os 3 modelos e o load balancer em 127.0.0.1:8000.
"""
import argparse
import csv
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

# .env fica na raiz do repositório (pai de experimentos/), independente do cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

OLLAMA = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
LB_URL = os.getenv("LB_URL", "http://127.0.0.1:8000").rstrip("/")
MODELO_BASELINE = os.getenv("FALLBACK_MODEL", "phi3.5")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
# Na proposta o servidor pode tentar o modelo escolhido e depois o fallback (2x timeout).
TIMEOUT = OLLAMA_TIMEOUT * 2 + 60
NUM_PREDICT = os.getenv("OLLAMA_NUM_PREDICT", "").strip()

COLUNAS = [
    "timestamp", "rodada", "passo", "ordem_exec", "cenario", "prompt_id", "classe_esperada", "classe",
    "modelo_escolhido", "modelo_atendeu", "desvio", "latencia_s", "lb_e2e_s",
    "resposta_chars", "sucesso", "erro",
]


def chamar_baseline(client, prompt):
    corpo = {"model": MODELO_BASELINE, "prompt": prompt, "stream": False}
    if NUM_PREDICT:
        corpo["options"] = {"num_predict": int(NUM_PREDICT)}
    r = client.post(f"{OLLAMA}/api/generate", json=corpo)
    r.raise_for_status()
    texto = (r.json().get("response") or "").strip()
    if not texto:
        raise ValueError("resposta vazia")  # o balancer também trata vazio como falha
    return {"classe": "", "modelo_escolhido": MODELO_BASELINE, "modelo_atendeu": MODELO_BASELINE,
            "desvio": False, "lb_e2e_s": "", "resposta_chars": len(texto)}


def chamar_proposta(client, prompt, prompt_id):
    r = client.post(f"{LB_URL}/infer", json={"prompt": prompt, "prompt_id": prompt_id})
    d = r.json()  # em falha o balancer devolve 502 com o JSON do despacho
    if r.status_code >= 400 or not d.get("ok", False):
        raise RuntimeError(f"http {r.status_code}: {d.get('error')}")
    return {"classe": d.get("label", ""), "modelo_escolhido": d.get("chosen_model", ""),
            "modelo_atendeu": d.get("served_model", ""), "desvio": d.get("redirected", ""),
            "lb_e2e_s": d.get("e2e_latency_s", ""), "resposta_chars": len(d.get("response") or "")}


def medir(cenario, client, item):
    t0 = time.perf_counter()
    try:
        info = (chamar_baseline(client, item["prompt"]) if cenario == "baseline"
                else chamar_proposta(client, item["prompt"], item["id"]))
        sucesso, erro = True, ""
    except Exception as e:  # falha conta como falha, não entra na latência
        info = {"classe": "", "modelo_escolhido": "", "modelo_atendeu": "", "desvio": "",
                "lb_e2e_s": "", "resposta_chars": ""}
        sucesso, erro = False, f"{type(e).__name__}: {e}"
    dt = time.perf_counter() - t0
    return {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "cenario": cenario, "prompt_id": item["id"],
            "classe_esperada": item["classe_esperada"], **info,
            "latencia_s": round(dt, 4), "sucesso": sucesso, "erro": erro}


def planejar_rodada(prompts, rodada, seed, evitar_inicio=None):
    """Devolve [(passo, item, cenario), ...] com o delineamento contrabalançado descrito no topo."""
    rng = random.Random(f"{seed}-{rodada}")
    primeiro = {}
    for classe in sorted({p["classe_esperada"] for p in prompts}):
        ids = [p["id"] for p in prompts if p["classe_esperada"] == classe]
        rng.shuffle(ids)
        for k, pid in enumerate(ids):
            primeiro[pid] = "baseline" if k % 2 == 0 else "proposta"
    passo1 = prompts[:]
    rng.shuffle(passo1)
    while evitar_inicio is not None and len(passo1) > 1 and passo1[0]["id"] == evitar_inicio:
        rng.shuffle(passo1)
    h = len(passo1) // 2
    passo2 = passo1[h:] + passo1[:h]  # cada prompt volta >= n/2 chamadas depois
    outro = {"baseline": "proposta", "proposta": "baseline"}
    return ([(1, p, primeiro[p["id"]]) for p in passo1] +
            [(2, p, outro[primeiro[p["id"]]]) for p in passo2])


def _get(client, url):
    try:
        r = client.get(url, timeout=15)
        return r.json() if r.status_code == 200 else {"_http": r.status_code}
    except Exception as e:  # coleta de metadados nunca derruba o experimento
        return {"_erro": f"{type(e).__name__}: {e}"}


def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return ""


def _ram_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        try:
            kb = int(next(l for l in open("/proc/meminfo") if l.startswith("MemTotal")).split()[1])
            return round(kb / 1e6, 1)
        except Exception:
            return None


def ambiente():
    gpu = ""
    if shutil.which("nvidia-smi"):
        gpu = _cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    raiz = Path(__file__).resolve().parent.parent
    return {
        "so": platform.platform(), "python": sys.version.split()[0],
        "cpu": (_cmd(["sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"]) or platform.processor()).strip(),
        "cpu_nucleos": os.cpu_count(), "ram_total_gb": _ram_gb(), "gpu": gpu or "não detectada por nvidia-smi",
        "git_commit": _cmd(["git", "-C", str(raiz), "rev-parse", "--short", "HEAD"]),
        "git_branch": _cmd(["git", "-C", str(raiz), "branch", "--show-current"]),
    }


def modelos_carregados(client):
    d = _get(client, f"{OLLAMA}/api/ps")
    return [{"nome": m.get("name"), "tamanho_vram_mb": round(m.get("size_vram", 0) / 1e6),
             "tamanho_mb": round(m.get("size", 0) / 1e6)} for m in d.get("models", [])]


def checar_prontidao(client):
    """Aborta antes de gastar a rodada se o ambiente não está pronto."""
    h = _get(client, f"{LB_URL}/health")
    if not h.get("ok"):
        sys.exit(f"ERRO: balancer não está pronto em {LB_URL}/health -> {h}\n"
                 "Suba o balancer em outro terminal: python -m src.loadbalancer")
    if not _get(client, f"{OLLAMA}/api/tags").get("models"):
        sys.exit(f"ERRO: Ollama sem modelos ou fora do ar em {OLLAMA}")
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(Path(__file__).resolve().parent / "prompts.json"))
    ap.add_argument("--saida", default=str(Path(__file__).resolve().parent / "resultados.csv"))
    ap.add_argument("--repeticoes", type=int, default=1, help="rodadas completas por cenário")
    ap.add_argument("--aquecimento", type=int, default=2, help="passadas descartadas")
    ap.add_argument("--seed", type=int, default=42, help="ordem dos prompts (mesma nos 2 cenários)")
    ap.add_argument("--anexar", action="store_true", help="permite acrescentar a um CSV existente")
    a = ap.parse_args()

    saida = Path(a.saida)
    if saida.exists() and saida.stat().st_size > 0 and not a.anexar:
        sys.exit(f"ERRO: {saida} já existe. Apague/renomeie (para não misturar rodadas) ou use --anexar.")

    prompts = json.loads(Path(a.prompts).read_text(encoding="utf-8"))

    print(f"Ollama={OLLAMA} balancer={LB_URL} baseline={MODELO_BASELINE} "
          f"num_predict={NUM_PREDICT or 'SEM LIMITE'} timeout_cliente_s={TIMEOUT}")
    if not NUM_PREDICT:
        print("AVISO: OLLAMA_NUM_PREDICT não definido; modelos podem gerar respostas de tamanhos muito diferentes.")

    meta = {"inicio_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "delineamento": "2 passos por rodada, contrabalançado por classe, sem prompt repetido em sequência",
            "argumentos": {"repeticoes": a.repeticoes, "aquecimento": a.aquecimento, "seed": a.seed,
                           "n_prompts": len(prompts)},
            "config": {"num_predict": NUM_PREDICT or None, "timeout_cliente_s": TIMEOUT,
                       "ollama_host": OLLAMA, "balancer": LB_URL, "modelo_baseline": MODELO_BASELINE},
            "ambiente": ambiente()}

    with httpx.Client(timeout=TIMEOUT) as client:
        meta["balancer_health"] = checar_prontidao(client)
        meta["ollama_versao"] = _get(client, f"{OLLAMA}/api/version").get("version")
        meta["modelos_instalados"] = [m.get("name") for m in _get(client, f"{OLLAMA}/api/tags").get("models", [])]

        # Aquecimento: carrega os modelos na memória; essas chamadas NÃO vão ao CSV.
        amostra = [next(p for p in prompts if p["classe_esperada"] == c)
                   for c in ("simples", "codigo", "complexo")]
        falhas_aq = 0
        for _ in range(a.aquecimento):
            for item in amostra:
                for cenario in ("baseline", "proposta"):
                    falhas_aq += 0 if medir(cenario, client, item)["sucesso"] else 1
        print(f"Aquecimento: {a.aquecimento} passada(s) x 3 prompts x 2 cenários (descartadas); falhas: {falhas_aq}")
        meta["aquecimento_falhas"] = falhas_aq
        meta["modelos_carregados_apos_aquecimento"] = modelos_carregados(client)
        print("Carregados no Ollama:", [m["nome"] for m in meta["modelos_carregados_apos_aquecimento"]])

        novo = not saida.exists() or saida.stat().st_size == 0
        with open(saida, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUNAS, lineterminator="\n")
            if novo:
                w.writeheader()
            # o último pedido do aquecimento é o último prompt da amostra: a 1a medição não pode repeti-lo
            ultimo_id = amostra[-1]["id"] if a.aquecimento > 0 else None
            for rodada in range(1, a.repeticoes + 1):
                plano_r = planejar_rodada(prompts, rodada, a.seed, evitar_inicio=ultimo_id)
                for ordem_exec, (passo, item, cenario) in enumerate(plano_r, 1):
                    linha = {"rodada": rodada, "passo": passo, "ordem_exec": ordem_exec,
                             **medir(cenario, client, item)}
                    w.writerow(linha)
                    f.flush()
                    print(f"[r{rodada} p{passo} #{ordem_exec:02d}] {cenario:8s} {item['id']} "
                          f"{linha['latencia_s']:.2f}s {'ok' if linha['sucesso'] else 'FALHA ' + linha['erro']}")
                ultimo_id = plano_r[-1][1]["id"]
        meta["modelos_carregados_ao_final"] = modelos_carregados(client)

    meta["fim_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    mpath = saida.with_name(saida.stem + ".metadados.json")
    mpath.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Metadados gravados em {mpath}")


if __name__ == "__main__":
    main()
