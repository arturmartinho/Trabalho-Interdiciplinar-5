"""Roda o experimento em uma matriz cenário × deployment e grava CSV v2.

Cenários:
  * ``default_model``: chamada direta ao Ollama com o mesmo ``model_tag``;
  * ``loadbalancer_classifier``: chamada a ``POST /infer`` (classificador +
    despacho + fallback existente).

Deployments:
  * ``local``: ``OLLAMA_HOST`` local (ou ``--ollama-local``);
  * ``remote``: ``OLLAMA_REMOTE_HOST`` (ou ``--ollama-remote``). O mesmo tag
    precisa existir nos dois hosts. O LB remoto é o ``--lb-remote-url``.

Medição: ``latency_e2e_s`` = ``time.perf_counter`` ao redor da chamada do
cenário. RAM/GPU são coletadas por ``ResourceSampler`` em background e
agregadas por bloco; token usage vem da resposta do Ollama (direta ou via LB).

Delineamento sequencial: contrabalançado entre todas as combinações da
matriz, com o mesmo prompt separado por pelo menos n/m chamadas (m =
combinações) e nenhuma repetição consecutiva. Modo carga (``--concorrencia``):
um bloco homogêneo por cenário×deployment, ordem alternada entre rodadas.

Uso:
  python -m src experiment --repeticoes 3 --aquecimento 2
  python experiments/rodar_experimento.py --prompts experiments/prompts_experimento.json \
      --scenarios default_model,loadbalancer_classifier --deployments local,remote
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import httpx

# Permite executar como script (``python experiments/rodar_experimento.py``)
# e como módulo: os imports de ``experiments`` e ``src`` precisam da raiz.
_RAIZ = Path(__file__).resolve().parent.parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

from experiments.analise_metricas import CENARIOS, CENARIO_LB  # noqa: E402
from experiments.config_experimento import (  # noqa: E402
    GPU_METRICS, LB_LOCAL_URL, LB_REMOTE_URL, LB_URL,
    NUM_PREDICT, OLLAMA_HOST, OLLAMA_REMOTE_HOST, RESOURCE_SAMPLE_MS, TIMEOUT,
    modelo_padrao,
)
from experiments.csv_resultados import COLUNAS, SCHEMA_VERSION, linha_vazia  # noqa: E402
from experiments.entrada import CLASSES, carregar_prompts  # noqa: E402
from experiments.recursos import ResourceSampler  # noqa: E402
# `chamar_baseline`/`chamar_proposta` também são reexportados para quem
# importava o runner antigo diretamente.
from experiments.runners import (  # noqa: E402
    ALIASES, ConfigExecucao, chamar_baseline, chamar_proposta,
    classificacao, normalizar_cenario, runner_para,
)
from src.loadbalancer.dispatcher import same_model  # noqa: E402

# Reexportados para compatibilidade com quem importava estas constantes.
OLLAMA = OLLAMA_HOST
DEPLOYMENTS = ("local", "remote")


def medir(
    cenario,
    client,
    item,
    *,
    cfg: ConfigExecucao | None = None,
    deployment: str = "local",
    runner=None,
    clock=time.perf_counter,
    run_id: str = "",
):
    """Executa um item e devolve a linha completa do CSV v2."""
    cfg = cfg or ConfigExecucao()
    nome = normalizar_cenario(cenario)
    runner = runner or runner_para(cenario, cfg, deployment)

    t0 = clock()
    try:
        info = runner.call(client, item, cfg, deployment)
        sucesso, erro = True, ""
    except Exception as exc:  # falha conta como falha; latência ainda é registrada
        info = {}
        sucesso, erro = False, f"{type(exc).__name__}: {exc}"
    dt = round(clock() - t0, 4)

    predita = str(info.get("classe") or "")
    return {
        **linha_vazia(),
        **info,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "cenario": cenario,
        "scenario": nome,
        "deployment": deployment,
        "prompt_id": item["id"],
        "classe_esperada": item["classe_esperada"],
        "model_tag": cfg.model_tag,
        "target_ollama_host": cfg.ollama_host(deployment),
        "target_lb_url": cfg.lb_url(deployment) if nome == CENARIO_LB else "",
        **cfg.effective(deployment, nome),
        "classe_predita": predita,
        **classificacao(nome, predita, item["classe_esperada"]),
        "latencia_s": dt,
        "latency_e2e_s": dt,
        "sucesso": sucesso,
        "erro": erro,
        "idioma": item.get("idioma", ""),
        "subtipo": item.get("subtipo", ""),
        "linguagem": item.get("linguagem", ""),
    }


def planejar_rodada(
    prompts,
    rodada,
    seed,
    evitar_inicio=None,
    cenarios=CENARIOS,
    deployments=("local",),
):
    """``[(passo, item, cenario, deployment), ...]`` contrabalançado.

    Cada prompt passa por todas as combinações cenário×deployment uma vez por
    rodada; quem vem primeiro é balanceado dentro de cada classe.
    """
    combinacoes = [(cenario, deployment) for deployment in deployments for cenario in cenarios]
    if not combinacoes:
        raise ValueError("é preciso selecionar ao menos um cenário e um deployment")
    rng = random.Random(f"{seed}-{rodada}")
    primeiro: dict[str, tuple[str, str]] = {}
    for classe in sorted({p["classe_esperada"] for p in prompts}):
        ids = [p["id"] for p in prompts if p["classe_esperada"] == classe]
        rng.shuffle(ids)
        for posicao, pid in enumerate(ids):
            primeiro[pid] = combinacoes[posicao % len(combinacoes)]

    ordem = prompts[:]
    rng.shuffle(ordem)
    while evitar_inicio is not None and len(ordem) > 1 and ordem[0]["id"] == evitar_inicio:
        rng.shuffle(ordem)

    n = len(ordem)
    distancia = max(1, n // len(combinacoes))
    plano = []
    for passo, _combinacao in enumerate(combinacoes):
        deslocamento = (passo * distancia) % n
        rotacionados = ordem[deslocamento:] + ordem[:deslocamento]
        for item in rotacionados:
            cenario, deployment = primeiro[item["id"]]
            combinacao = combinacoes[(combinacoes.index((cenario, deployment)) + passo) % len(combinacoes)]
            plano.append((passo + 1, item, combinacao[0], combinacao[1]))
    return plano


def planejar_carga_matriz(prompts, rodada, seed, cenarios=CENARIOS, deployments=("local",)):
    """``[(passo, cenario, deployment, itens), ...]``; blocos homogêneos."""
    rng = random.Random(f"{seed}-carga-{rodada}")
    ordem = prompts[:]
    rng.shuffle(ordem)
    combinacoes = [(cenario, deployment) for deployment in deployments for cenario in cenarios]
    if rodada % 2 == 0:
        combinacoes = list(reversed(combinacoes))
    return [
        (passo, cenario, deployment, ordem)
        for passo, (cenario, deployment) in enumerate(combinacoes, 1)
    ]


def planejar_carga(
    prompts,
    rodada,
    seed,
    cenarios=("baseline", "proposta"),
    deployments=("local",),
):
    """Compatibilidade com o chamador antigo (um deployment → 3-tuplas)."""
    blocos = planejar_carga_matriz(prompts, rodada, seed, cenarios, deployments)
    if len(deployments) == 1:
        return [(passo, cenario, itens) for passo, cenario, _dep, itens in blocos]
    return blocos


def rodar_bloco(
    client,
    itens,
    cenario,
    concorrencia,
    ao_terminar,
    *,
    medir_fn=None,
    sampler: ResourceSampler | None = None,
    deployment: str = "local",
):
    """Executa itens com ``concorrencia`` requisições simultâneas.

    Com ``sampler``, as linhas são entregues a ``ao_terminar`` depois de
    ``stop()``, já enriquecidas com média/pico de RAM e GPU do bloco.
    """
    funcao = medir_fn or medir
    pendentes = []
    if sampler is not None:
        sampler.reset()
        sampler.start()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concorrencia) as pool:
        futuros = [
            (k, item, pool.submit(funcao, cenario, client, item))
            for k, item in enumerate(itens, 1)
        ]
        for k, item, futuro in futuros:
            linha = dict(futuro.result())
            if sampler is None:
                ao_terminar(k, item, linha)
            else:
                pendentes.append((k, item, linha))
    duracao = time.perf_counter() - t0
    recursos = sampler.stop() if sampler is not None else {}
    for k, item, linha in pendentes:
        ao_terminar(k, item, {**linha, **recursos})
    return duracao


# --------------------------------------------------------------------------- #
# Metadados e checagens
# --------------------------------------------------------------------------- #
def _get(client, url):
    try:
        resposta = client.get(url, timeout=15)
        return resposta.json() if resposta.status_code == 200 else {"_http": resposta.status_code}
    except Exception as exc:  # metadados nunca derrubam a medição
        return {"_erro": f"{type(exc).__name__}: {exc}"}


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
        "cpu_nucleos": os.cpu_count(), "ram_total_gb": _ram_gb(),
        "gpu": gpu or "não detectada por nvidia-smi",
        "git_commit": _cmd(["git", "-C", str(raiz), "rev-parse", "--short", "HEAD"]),
        "git_branch": _cmd(["git", "-C", str(raiz), "branch", "--show-current"]),
    }


def modelos_carregados(client, ollama_host: str | None = None):
    host = (ollama_host or OLLAMA_HOST).rstrip("/")
    dados = _get(client, f"{host}/api/ps")
    return [{"nome": m.get("name"), "tamanho_vram_mb": round(m.get("size_vram", 0) / 1e6),
             "tamanho_mb": round(m.get("size", 0) / 1e6)} for m in dados.get("models", [])]


def checar_deployments(client, cfg, deployments, cenarios):
    """Pré-checa ``/api/tags``, ``/api/ps`` e ``/health`` por deployment."""
    checagens = {}
    for deployment in deployments:
        host = cfg.ollama_host(deployment)
        tags = _get(client, f"{host}/api/tags")
        ps = _get(client, f"{host}/api/ps")
        health = _get(client, f"{cfg.lb_url(deployment)}/health") if CENARIO_LB in cenarios else {"_skip": True}
        instalados = [m.get("name") for m in tags.get("models", [])]
        falhas = []
        if tags.get("_erro") or tags.get("_http") or not instalados:
            falhas.append(f"Ollama {host} sem modelos ou fora do ar ({tags})")
        if CENARIO_LB in cenarios and not health.get("ok"):
            falhas.append(f"balancer fora do ar em {cfg.lb_url(deployment)}/health ({health})")
        # Ollama aceita `phi3.5` como atalho de `phi3.5:latest`; comparar a
        # família evita abortar uma rodada válida só por causa do tag.
        presente = any(same_model(str(nome), cfg.model_tag) for nome in instalados if nome)
        if not presente and instalados:
            falhas.append(f"modelo {cfg.model_tag} não está em {host}/api/tags")
        checagens[deployment] = {
            "ollama_host": host,
            "lb_url": cfg.lb_url(deployment) if CENARIO_LB in cenarios else "",
            "api_tags": tags, "api_ps": ps, "health": health,
            "modelos_instalados": instalados,
            "model_tag_solicitado": cfg.model_tag,
            "model_tag_presente": presente,
            "falhas": falhas,
        }
        if falhas:
            sys.exit(f"ERRO: deployment '{deployment}': " + "; ".join(falhas))
    return checagens


def checar_prontidao(client):
    """Compatibilidade com a checagem de um único deployment."""
    health = _get(client, f"{LB_URL}/health")
    if not health.get("ok"):
        sys.exit(f"ERRO: balancer não está pronto em {LB_URL}/health -> {health}\n"
                 "Suba o balancer em outro terminal: python -m src serve")
    if not _get(client, f"{OLLAMA_HOST}/api/tags").get("models"):
        sys.exit(f"ERRO: Ollama sem modelos ou fora do ar em {OLLAMA_HOST}")
    return health


def _parse_opcoes(valor: str, permitidos, nome: str):
    itens = [parte.strip().lower() for parte in str(valor).split(",") if parte.strip()]
    if not itens:
        raise ValueError(f"--{nome} precisa de ao menos um valor")
    normalizados = []
    for item in itens:
        item = ALIASES.get(item, item)
        if item not in permitidos:
            raise ValueError(f"--{nome}: '{item}' inválido (use {', '.join(permitidos)})")
        if item not in normalizados:
            normalizados.append(item)
    return tuple(normalizados)


def _construir_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(Path(__file__).resolve().parent / "prompts.json"))
    ap.add_argument("--saida", default=str(Path(__file__).resolve().parent / "resultados.csv"))
    ap.add_argument("--repeticoes", type=int, default=1, help="rodadas completas da matriz")
    ap.add_argument("--aquecimento", type=int, default=2, help="passadas descartadas")
    ap.add_argument("--seed", type=int, default=42, help="ordem dos prompts (mesma nos cenários)")
    ap.add_argument("--anexar", action="store_true", help="permite acrescentar a um CSV existente")
    ap.add_argument("--concorrencia", type=int, default=1,
                    help="requisições simultâneas (1 = contrabalançado; >1 = modo carga)")
    ap.add_argument("--scenarios", "--cenarios", default=",".join(CENARIOS),
                    help="default_model,loadbalancer_classifier (aceita baseline,proposta)")
    ap.add_argument("--deployments", default="local", help="local,remote")
    ap.add_argument("--model-tag", default=None,
                    help="tag usado no default_model nos dois deployments "
                         "(padrão: maior modelo registrado no catálogo do LB)")
    ap.add_argument("--ollama-local", default=OLLAMA_HOST)
    ap.add_argument("--ollama-remote", default=OLLAMA_REMOTE_HOST)
    ap.add_argument("--lb-local-url", default=LB_LOCAL_URL)
    ap.add_argument("--lb-remote-url", default=LB_REMOTE_URL)
    ap.add_argument("--resource-sample-ms", type=int, default=RESOURCE_SAMPLE_MS,
                    help="intervalo do sampler de RAM/GPU (background)")
    ap.add_argument("--gpu-metrics", choices=("auto", "off", "nvidia-smi"), default=GPU_METRICS)
    return ap


def main():
    args = _construir_parser().parse_args()
    if args.concorrencia < 1:
        sys.exit("ERRO: --concorrencia precisa ser >= 1")
    if args.aquecimento < 0 or args.repeticoes < 1:
        sys.exit("ERRO: --aquecimento precisa ser >= 0 e --repeticoes >= 1")
    try:
        cenarios = _parse_opcoes(args.scenarios, CENARIOS, "scenarios")
        deployments = _parse_opcoes(args.deployments, DEPLOYMENTS, "deployments")
        cfg = ConfigExecucao(
            model_tag=(args.model_tag or "").strip() or modelo_padrao(),
            ollama_local=args.ollama_local,
            ollama_remote=args.ollama_remote,
            lb_local_url=args.lb_local_url,
            lb_remote_url=args.lb_remote_url,
            num_predict=NUM_PREDICT,
        )
        prompts = carregar_prompts(args.prompts)
    except ValueError as exc:
        sys.exit(f"ERRO: {exc}")

    if "remote" in deployments and not cfg.ollama_remote:
        sys.exit("ERRO: --deployments remote exige --ollama-remote ou OLLAMA_REMOTE_HOST")
    if "remote" in deployments and CENARIO_LB in cenarios and not cfg.lb_remote_url:
        sys.exit("ERRO: --deployments remote com loadbalancer_classifier exige --lb-remote-url ou LB_REMOTE_URL")

    saida = Path(args.saida)
    if saida.exists() and saida.stat().st_size > 0 and not args.anexar:
        sys.exit(f"ERRO: {saida} já existe. Apague/renomeie (para não misturar rodadas) ou use --anexar.")

    print(f"cenários={','.join(cenarios)} deployments={','.join(deployments)} "
          f"model_tag={cfg.model_tag} num_predict={NUM_PREDICT or 'SEM LIMITE'} "
          f"timeout_cliente_s={TIMEOUT}")
    if not NUM_PREDICT:
        print("AVISO: OLLAMA_NUM_PREDICT não definido; modelos podem gerar respostas de tamanhos muito diferentes.")

    run_id = uuid.uuid4().hex[:12]
    matriz = [
        {"scenario": cenario, "deployment": deployment, **cfg.effective(deployment, cenario)}
        for deployment in deployments for cenario in cenarios
    ]
    delineamento = (
        f"{len(matriz)} combinações por rodada, contrabalançadas por classe, sem prompt repetido em sequência"
        if args.concorrencia == 1 else
        f"modo carga: {len(matriz)} blocos por rodada (um por cenário×deployment, ordem alternada), "
        f"{args.concorrencia} simultâneas"
    )
    meta = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "inicio_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "delineamento": delineamento,
        "matriz": matriz,
        "argumentos": {
            "repeticoes": args.repeticoes, "aquecimento": args.aquecimento,
            "seed": args.seed, "n_prompts": len(prompts), "concorrencia": args.concorrencia,
            "prompts": Path(args.prompts).name, "scenarios": list(cenarios),
            "deployments": list(deployments),
        },
        "config": {
            "num_predict": NUM_PREDICT or None, "timeout_cliente_s": TIMEOUT,
            "model_tag": cfg.model_tag,
            "ollama_local": cfg.ollama_local, "ollama_remote": cfg.ollama_remote,
            "lb_local_url": cfg.lb_local_url, "lb_remote_url": cfg.lb_remote_url,
            "resource_sample_ms": args.resource_sample_ms, "gpu_metrics": args.gpu_metrics,
        },
        "regras_medicao": {
            "latencia": "latency_e2e_s = perf_counter ao redor da chamada do cenário (ponta a ponta no cliente)",
            "ram": "psutil: RSS do processo e RAM do sistema (used MB / percent); avg e peak por bloco, cache do último sample válido",
            "gpu": "nvidia-smi por amostra: utilização = média das GPUs; memória e potência = soma; pico = máximo entre amostras",
            "tokens": "prompt_tokens=prompt_eval_count, completion_tokens=eval_count, total=soma; durações ns→s",
            "loss": "classification_error = 0/1 apenas em loadbalancer_classifier; default_model é NA e não entra no denominador",
        },
        "ambiente": ambiente(),
        "carga": [],
    }

    with httpx.Client(timeout=TIMEOUT) as client:
        meta["checagens"] = checar_deployments(client, cfg, deployments, cenarios)
        meta["ollama_versao"] = _get(client, f"{cfg.ollama_local}/api/version").get("version")

        # Aquecimento: carrega os modelos; estas chamadas não vão ao CSV.
        amostra = []
        for classe in CLASSES:
            candidato = next((p for p in prompts if p["classe_esperada"] == classe), None)
            if candidato:
                amostra.append(candidato)
        falhas_aq = 0
        for deployment in deployments:
            for cenario in cenarios:
                runner = runner_para(cenario, cfg, deployment)
                for _ in range(args.aquecimento):
                    for item in amostra:
                        falhas_aq += 0 if medir(cenario, client, item, cfg=cfg,
                                               deployment=deployment, runner=runner,
                                               run_id=run_id)["sucesso"] else 1
        print(f"Aquecimento: {args.aquecimento} passada(s) x {len(amostra)} prompts x {len(matriz)} combinações "
              f"(descartadas); falhas: {falhas_aq}")
        meta["aquecimento_falhas"] = falhas_aq
        meta["modelos_carregados_apos_aquecimento"] = modelos_carregados(client, cfg.ollama_local)
        print("Carregados no Ollama local:", [m["nome"] for m in meta["modelos_carregados_apos_aquecimento"]])

        novo = not saida.exists() or saida.stat().st_size == 0
        trava = threading.Lock()
        with open(saida, "a", newline="", encoding="utf-8") as arquivo:
            escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS, lineterminator="\n")
            if novo:
                escritor.writeheader()
            sampler = ResourceSampler(args.resource_sample_ms, args.gpu_metrics)

            for rodada in range(1, args.repeticoes + 1):
                if args.concorrencia > 1:
                    for passo, cenario, deployment, itens in planejar_carga_matriz(
                        prompts, rodada, args.seed, cenarios, deployments
                    ):
                        medidor = partial(medir, cfg=cfg, deployment=deployment, run_id=run_id)

                        def registrar(k, item, linha, *, rodada=rodada, passo=passo,
                                      cenario=cenario, deployment=deployment):
                            completa = {"rodada": rodada, "passo": passo, "ordem_exec": k,
                                        "concorrencia": args.concorrencia, **linha}
                            with trava:
                                escritor.writerow(completa)
                                arquivo.flush()
                            print(f"[r{rodada} p{passo} #{k:03d}] {deployment}/{cenario} {item['id']} "
                                  f"{completa['latencia_s']:.2f}s "
                                  f"{'ok' if completa['sucesso'] else 'FALHA ' + completa['erro']}")

                        dur = rodar_bloco(client, itens, cenario, args.concorrencia, registrar,
                                          medir_fn=medidor, sampler=sampler, deployment=deployment)
                        meta["carga"].append({
                            "rodada": rodada, "passo": passo, "cenario": cenario,
                            "deployment": deployment, "n": len(itens),
                            "concorrencia": args.concorrencia, "duracao_s": round(dur, 3),
                            "vazao_rps": round(len(itens) / dur, 4) if dur else None,
                        })
                        print(f"Bloco r{rodada} {deployment}/{cenario}: {len(itens)} req em {dur:.1f}s "
                              f"({len(itens) / dur:.3f} req/s)" if dur else "")

                else:
                    plano = planejar_rodada(prompts, rodada, args.seed, cenarios=cenarios,
                                            deployments=deployments)
                    itens_ctx = [{**item, "_cenario": cenario, "_deployment": deployment,
                                  "_passo": passo} for passo, item, cenario, deployment in plano]

                    def medir_sequencial(_cenario, _client, item):
                        return medir(item["_cenario"], _client, item, cfg=cfg,
                                     deployment=item["_deployment"], run_id=run_id)

                    def registrar_sequencial(k, item, linha, *, rodada=rodada):
                        completa = {"rodada": rodada, "passo": item["_passo"], "ordem_exec": k,
                                    "concorrencia": 1, **linha}
                        with trava:
                            escritor.writerow(completa)
                            arquivo.flush()
                        print(f"[r{rodada} p{item['_passo']} #{k:03d}] "
                              f"{item['_deployment']}/{item['_cenario']} {item['id']} "
                              f"{completa['latencia_s']:.2f}s "
                              f"{'ok' if completa['sucesso'] else 'FALHA ' + completa['erro']}")

                    dur = rodar_bloco(client, itens_ctx, None, 1, registrar_sequencial,
                                      medir_fn=medir_sequencial, sampler=sampler)
                    meta["carga"].append({
                        "rodada": rodada, "modo": "sequencial", "n": len(itens_ctx),
                        "concorrencia": 1, "duracao_s": round(dur, 3),
                        "vazao_rps": round(len(itens_ctx) / dur, 4) if dur else None,
                    })

        meta["resource_sampler"] = {
            "interval_ms": args.resource_sample_ms, "gpu_metrics": args.gpu_metrics,
            "regra_multi_gpu": "utilização média; memória e potência somadas",
            "resumo_final": sampler.resumo(),
        }
        meta["modelos_carregados_ao_final"] = modelos_carregados(client, cfg.ollama_local)

    meta["fim_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    caminho_meta = saida.with_name(saida.stem + ".metadados.json")
    caminho_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Metadados gravados em {caminho_meta}")


if __name__ == "__main__":
    main()
