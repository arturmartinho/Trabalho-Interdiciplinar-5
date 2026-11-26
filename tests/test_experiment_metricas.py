"""Testes focados das métricas, do sampler e do schema v2 dos experimentos."""

import csv
import importlib.util
import json
import sys
import time
from itertools import product
from pathlib import Path

import pytest

from experiments import (
    analisar, analise_metricas, config_experimento, csv_resultados, entrada,
    recursos, runners,
)
from experiments.analise_metricas import (
    CENARIO_LB, CENARIO_PADRAO, CENARIOS, agregar, metricas,
)
from experiments.csv_resultados import COLUNAS_V1, COLUNAS_V2, linha_vazia
from experiments.recursos import ResourceSampler
from experiments.runners import ConfigExecucao, classificacao

EXP = Path(__file__).resolve().parents[1] / "experiments"


def _importar(nome):
    import importlib.util
    spec = importlib.util.spec_from_file_location(nome, EXP / f"{nome}.py")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nome] = modulo
    spec.loader.exec_module(modulo)
    return modulo


rx = _importar("rodar_experimento")


class _Memoria:
    used = 2048 * 1024 * 1024
    percent = 40.0


class _Processo:
    def memory_info(self):
        return type("Info", (), {"rss": 512 * 1024 * 1024})()


class _Psutil:
    def Process(self):
        return _Processo()

    def virtual_memory(self):
        return _Memoria()


def _gpu_dupla():
    return [
        {"util_pct": 10, "mem_used_mb": 100, "mem_total_mb": 1000, "power_w": 20},
        {"util_pct": 30, "mem_used_mb": 200, "mem_total_mb": 1200, "power_w": 40},
    ]


# --------------------------------------------------------------------------- #
# Validação e matriz
# --------------------------------------------------------------------------- #
def test_carregar_prompts_valida_e_normaliza(tmp_path):
    origem = tmp_path / "prompts.json"
    origem.write_text(json.dumps([
        {"id": " p1 ", "prompt": "  oi  ", "classe_esperada": " Simples ", "idioma": "pt"},
        {"id": "p2", "prompt": "x", "classe_esperada": "codigo"},
    ]), encoding="utf-8")
    itens = entrada.carregar_prompts(origem)
    assert itens[0]["id"] == "p1" and itens[0]["prompt"] == "oi"
    assert itens[0]["classe_esperada"] == "simples"
    assert itens[1]["idioma"] == ""  # metadata ausente vira string vazia

    for invalido in (
        [{"id": "p", "prompt": "oi", "classe_esperada": "outra"}],
        [{"id": "", "prompt": "oi", "classe_esperada": "simples"}],
        [{"id": "p", "classe_esperada": "simples"}],
        [{"id": "p", "prompt": "a", "classe_esperada": "simples"},
         {"id": "p", "prompt": "b", "classe_esperada": "simples"}],
        {"id": "p"},
    ):
        origem.write_text(json.dumps(invalido), encoding="utf-8")
        with pytest.raises(ValueError):
            entrada.carregar_prompts(origem)


def test_plano_cobre_matriz_e_nao_repete_prompt_em_sequencia():
    prompts = [{"id": f"P{i}", "classe_esperada": "simples", "prompt": "x"} for i in range(8)]
    plano = rx.planejar_rodada(prompts, 1, 42, cenarios=CENARIOS, deployments=("local", "remote"))
    assert len(plano) == 8 * len(CENARIOS) * 2
    vistos = {}
    for passo, item, cenario, deployment in plano:
        vistos.setdefault(item["id"], set()).add((cenario, deployment))
    assert all(par == set(product(CENARIOS, ("local", "remote"))) for par in vistos.values())
    assert all(a[1]["id"] != b[1]["id"] for a, b in zip(plano, plano[1:]))


def test_classificacao_e_loss_apenas_no_lb():
    assert classificacao(CENARIO_PADRAO, "", "simples") == {
        "classification_correct": "", "classification_error": ""}
    assert classificacao(CENARIO_LB, "simples", "simples")["classification_error"] == 0
    assert classificacao(CENARIO_LB, "codigo", "simples")["classification_error"] == 1


class _Resposta:
    status_code = 200

    def __init__(self, dados):
        self._dados = dados

    def raise_for_status(self):
        return None

    def json(self):
        return self._dados


class _Cliente:
    def __init__(self, dados):
        self.dados = dados
        self.chamadas = []

    def post(self, url, json):
        self.chamadas.append((url, json))
        return _Resposta(self.dados)


def test_medir_default_model_spelha_tokens_e_nao_classifica():
    cfg = ConfigExecucao(model_tag="phi3.5", ollama_local="http://local:11434")
    cliente = _Cliente({"response": "Paris", "prompt_eval_count": 8, "eval_count": 2,
                        "eval_duration": 1_000_000_000})
    item = {"id": "p1", "prompt": "capital?", "classe_esperada": "simples"}
    linha = rx.medir(CENARIO_PADRAO, cliente, item, cfg=cfg, run_id="run1")

    assert linha["schema_version"] == "2" and linha["run_id"] == "run1"
    assert linha["scenario"] == CENARIO_PADRAO and linha["deployment"] == "local"
    assert linha["model_tag"] == "phi3.5" and linha["prompt_tokens"] == 8
    assert linha["total_tokens"] == 10 and linha["eval_duration_s"] == 1.0
    assert linha["classe_predita"] == "" and linha["classification_error"] == ""
    assert linha["latency_e2e_s"] == linha["latencia_s"]
    assert cliente.chamadas[0][0] == "http://local:11434/api/generate"


def test_medir_lb_marca_usage_indisponivel_e_erro_de_classe():
    cfg = ConfigExecucao(model_tag="phi3.5", lb_local_url="http://lb:8000")
    cliente = _Cliente({"ok": True, "label": "codigo", "chosen_model": "qwen",
                        "served_model": "qwen", "redirected": False,
                        "response": "print(1)", "e2e_latency_s": 0.1})
    item = {"id": "p2", "prompt": "código", "classe_esperada": "simples"}
    linha = rx.medir(CENARIO_LB, cliente, item, cfg=cfg, run_id="run2")

    assert linha["token_source"] == "lb_unavailable"
    assert linha["classe_predita"] == "codigo" and linha["classification_error"] == 1
    assert linha["lb_e2e_s"] == 0.1
    assert cliente.chamadas[0][0] == "http://lb:8000/infer"


def test_maior_modelo_registrado_usa_catalogo_do_lb(monkeypatch):
    catalogo = [
        {"name": "pequeno:1b", "status": "up", "url": "http://a:11434"},
        {"name": "grande:27b", "status": "up", "url": "http://b:11434"},
        # down não é "disponível", mesmo sendo maior
        {"name": "desligado:100b", "status": "down", "url": "http://b:11434"},
    ]
    tags_por_host = {
        "http://a:11434": [
            {"name": "pequeno:1b:latest", "size": 1e9,
             "details": {"parameter_size": "1.0B"}},
        ],
        "http://b:11434": [
            {"name": "grande:27b:latest", "size": 17e9,
             "details": {"parameter_size": "27.3B"}},
            {"name": "desligado:100b", "size": 99e9,
             "details": {"parameter_size": "100B"}},
        ],
    }
    monkeypatch.setattr(config_experimento, "_catalogo", lambda: catalogo)
    monkeypatch.setattr(config_experimento, "_tags", lambda host: tags_por_host.get(host, []))

    assert config_experimento.maior_modelo_registrado() == "grande:27b"


def test_modelo_padrao_respeita_env_e_uso_sem_flag(monkeypatch):
    monkeypatch.setattr(config_experimento, "maior_modelo_registrado", lambda: "grande:27b")
    monkeypatch.delenv("MODELO_BASELINE", raising=False)
    assert config_experimento.modelo_padrao() == "grande:27b"

    monkeypatch.setattr(runners, "modelo_padrao", config_experimento.modelo_padrao)
    assert ConfigExecucao().model_tag == "grande:27b"
    assert ConfigExecucao(model_tag="fixado:8b").model_tag == "fixado:8b"

    monkeypatch.setenv("MODELO_BASELINE", "fixado:8b")
    assert config_experimento.modelo_padrao() == "fixado:8b"

    monkeypatch.setattr(config_experimento, "maior_modelo_registrado", lambda: "")
    monkeypatch.delenv("MODELO_BASELINE", raising=False)
    assert config_experimento.modelo_padrao() == config_experimento.MODELO_FALLBACK


def test_csv_v2_mantem_colunas_antigas_e_campos_novos():
    assert COLUNAS_V2[:len(COLUNAS_V1)] == COLUNAS_V1
    assert len(set(COLUNAS_V2)) == len(COLUNAS_V2)
    for campo in ("schema_version", "scenario", "deployment", "classification_error",
                  "latency_e2e_s", "total_tokens", "ram_rss_mb_peak", "gpu_power_w_peak"):
        assert campo in COLUNAS_V2
    assert set(linha_vazia()) == set(COLUNAS_V2)


# --------------------------------------------------------------------------- #
# Sampler
# --------------------------------------------------------------------------- #
def test_sampler_agrega_multi_gpu_e_ram():
    sampler = ResourceSampler(interval_ms=1, gpu_metrics="nvidia-smi",
                              psutil_module=_Psutil(), gpu_query=_gpu_dupla)
    sampler.start()
    time.sleep(0.02)
    resumo = sampler.stop()
    assert resumo["gpu_available"] is True and resumo["gpu_count"] == 2
    assert resumo["gpu_util_pct_avg"] == 20.0
    assert resumo["gpu_mem_used_mb_avg"] == 300.0
    assert resumo["gpu_mem_total_mb"] == 2200.0
    assert resumo["gpu_power_w_avg"] == 60.0
    assert resumo["gpu_sample_count"] >= 1
    assert resumo["ram_rss_mb_avg"] == 512.0
    assert resumo["ram_system_pct_peak"] == 40.0
    assert resumo["gpu_sample_interval_ms"] == 1


def test_sampler_reaproveita_cache_quando_gpu_falha():
    estado = {"falha": False}

    def gpu():
        if estado["falha"]:
            raise RuntimeError("nvidia-smi indisponível")
        return [{"util_pct": 50, "mem_used_mb": 10, "mem_total_mb": 100, "power_w": 5}]

    sampler = ResourceSampler(interval_ms=1, gpu_metrics="nvidia-smi",
                              psutil_module=_Psutil(), gpu_query=gpu)
    primeira = sampler.coletar_amostra()
    estado["falha"] = True
    segunda = sampler.coletar_amostra()
    assert primeira["gpu_util_pct"] == segunda["gpu_util_pct"] == 50
    assert "gpu:" in sampler.resumo()["resource_error"]


def test_sampler_cpu_only_deixa_campos_numericos_vazios():
    sampler = ResourceSampler(interval_ms=1, gpu_metrics="off", psutil_module=_Psutil())
    amostra = sampler.coletar_amostra()
    resumo = sampler.resumo()
    assert "gpu_util_pct" not in amostra
    assert resumo["gpu_available"] is False and resumo["gpu_count"] == 0
    assert resumo["gpu_util_pct_avg"] is None and resumo["gpu_mem_total_mb"] is None
    assert resumo["ram_rss_mb_avg"] == 512.0


def test_sampler_sem_psutil_marca_erro_sem_derrubar_medicao():
    class SemPsutil:
        def Process(self):
            raise ImportError("psutil ausente")

    sampler = ResourceSampler(interval_ms=1, gpu_metrics="off", psutil_module=SemPsutil())
    sampler.coletar_amostra()
    resumo = sampler.resumo()
    assert resumo["ram_rss_mb_avg"] is None
    assert "psutil" in resumo["resource_error"]


# --------------------------------------------------------------------------- #
# Agregação e relatório
# --------------------------------------------------------------------------- #
def _linha(cenario, deployment, classe, lat, erro, *, predita="simples", v2=True, **extras):
    base = {
        "cenario": cenario, "deployment": deployment, "classe_esperada": classe,
        "sucesso": "True", "latencia_s": lat, "classe": predita,
        "resposta_chars": "10",
    }
    if v2:
        base.update({
            "schema_version": "2", "scenario": cenario,
            "classification_error": erro, "prompt_tokens": "5", "total_tokens": "8",
            "ram_rss_mb_avg": "100", "ram_rss_mb_peak": "120",
            "gpu_available": "True" if extras.get("gpu_available") else "False",
            "gpu_util_pct_avg": "30" if extras.get("gpu_available") else "",
        })
    return base


def test_metricas_e_agregacao_p95_p99_e_error_rate():
    assert metricas([1, 2, 3, 4])["p95"] == pytest.approx(3.85)
    linhas = [
        _linha(CENARIO_LB, "local", "simples", 1, 0),
        _linha(CENARIO_LB, "local", "simples", 2, 1, predita="codigo"),
        _linha(CENARIO_LB, "local", "simples", 3, 0),
        _linha(CENARIO_PADRAO, "local", "simples", 4, ""),
        _linha("baseline", "remote", "codigo", 5, "", v2=False),
    ]
    grupos = agregar(linhas)
    lb = grupos[(CENARIO_LB, "local", "simples")]
    assert lb["n"] == 3 and lb["classification_error_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert lb["n_classified"] == 3 and lb["latency_p95_s"] == pytest.approx(2.9)
    assert lb["prompt_tokens_avg"] == 5 and lb["total_tokens_sum"] == 24
    assert grupos[(CENARIO_PADRAO, "local", "todos")]["n_classified"] == 0
    legado = grupos[(CENARIO_PADRAO, "remote", "codigo")]
    assert legado["n"] == 1 and legado["latency_mean_s"] == 5
    assert grupos[("todos", "todos", "todos")]["n"] == 5


def test_main_escreve_csv_v2_da_matriz_com_cliente_falso(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps([
        {"id": "p1", "prompt": "capital?", "classe_esperada": "simples", "idioma": "pt"},
    ]), encoding="utf-8")
    saida = tmp_path / "resultados.csv"

    class _ClienteHTTP:
        def get(self, url, **_kwargs):
            if url.endswith("/api/tags"):
                return _Resposta({"models": [{"name": "phi3.5"}]})
            if url.endswith("/api/ps"):
                return _Resposta({"models": []})
            if url.endswith("/api/version"):
                return _Resposta({"version": "teste"})
            return _Resposta({"ok": True})

        def post(self, url, json):
            if url.endswith("/api/generate"):
                return _Resposta({"response": "Paris", "prompt_eval_count": 4, "eval_count": 2})
            return _Resposta({"ok": True, "label": "simples", "chosen_model": "phi3.5",
                              "served_model": "phi3.5", "redirected": False,
                              "response": "Paris", "e2e_latency_s": 0.1})

    class _Contexto:
        def __enter__(self):
            return _ClienteHTTP()

        def __exit__(self, *_exc):
            return None

    monkeypatch.setattr(rx.httpx, "Client", lambda **_kwargs: _Contexto())
    monkeypatch.setattr(sys, "argv", [
        "rodar_experimento.py", "--prompts", str(prompts), "--saida", str(saida),
        "--repeticoes", "1", "--aquecimento", "0", "--gpu-metrics", "off",
        "--scenarios", "default_model,loadbalancer_classifier",
        "--deployments", "local,remote", "--ollama-remote", "http://remoto:11434",
        "--lb-remote-url", "http://remoto:8000", "--model-tag", "phi3.5",
    ])
    rx.main()

    with open(saida, encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))
    assert len(linhas) == 4
    assert {(r["scenario"], r["deployment"]) for r in linhas} == set(product(CENARIOS, ("local", "remote")))
    assert {r["model_tag"] for r in linhas} == {"phi3.5"}
    assert all(r["schema_version"] == "2" and "ram_rss_mb_avg" in r for r in linhas)
    assert all(r["gpu_available"] == "False" for r in linhas)
    assert all(r["classification_error"] == "" for r in linhas if r["scenario"] == CENARIO_PADRAO)
    assert all(r["classification_error"] == "0" for r in linhas if r["scenario"] == CENARIO_LB)

    meta = json.loads(saida.with_name("resultados.metadados.json").read_text(encoding="utf-8"))
    assert len(meta["matriz"]) == 4
    assert meta["checagens"]["remote"]["ollama_host"] == "http://remoto:11434"
    assert meta["regras_medicao"]["loss"].startswith("classification_error")


def test_analise_gera_resumo_e_relatorio(tmp_path, monkeypatch):
    entrada_csv = tmp_path / "resultados.csv"
    with open(entrada_csv, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS_V2, lineterminator="\n")
        escritor.writeheader()
        for cenario, lat, erro in ((CENARIO_PADRAO, 1.0, ""), (CENARIO_LB, 0.5, 0)):
            linha = linha_vazia()
            linha.update({"sucesso": "True", "latencia_s": lat, "latency_e2e_s": lat,
                          "cenario": cenario, "scenario": cenario, "deployment": "local",
                          "classe_esperada": "simples", "classe": "simples",
                          "classe_predita": "simples" if cenario == CENARIO_LB else "",
                          "classification_error": erro, "prompt_id": "p1", "resposta_chars": "10"})
            escritor.writerow(linha)

    resumo = tmp_path / "resumo.csv"
    relatorio = tmp_path / "relatorio.md"
    monkeypatch.setattr(sys, "argv", ["analisar.py", "--entrada", str(entrada_csv),
                                      "--pasta", str(tmp_path / "graficos"),
                                      "--resumo", str(resumo), "--relatorio", str(relatorio)])
    analisar.main()

    with open(resumo, encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))
    assert {"latency_p95_s", "latency_p99_s", "classification_error_rate"} <= set(linhas[0])
    texto = relatorio.read_text(encoding="utf-8")
    assert CENARIO_PADRAO in texto and "taxa de erro de classificação" in texto
    assert "não entra" in texto
    if importlib.util.find_spec("matplotlib"):
        assert (tmp_path / "graficos" / "grafico1_latencia_geral.png").exists()
