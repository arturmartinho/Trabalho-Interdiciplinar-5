"""Testes dos scripts de experimento que não precisam de Ollama."""

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments"


def _importar(nome):
    spec = importlib.util.spec_from_file_location(nome, EXP / f"{nome}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nome] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gp():
    return _importar("gerar_prompts")


@pytest.fixture(scope="module")
def rx():
    return _importar("rodar_experimento")


# --------------------------------------------------------------------------- #
# gerar_prompts
# --------------------------------------------------------------------------- #
def test_celulas_deterministicas_e_cobrem_a_matriz(gp):
    a, b = gp.montar_celulas(42), gp.montar_celulas(42)
    assert a == b
    assert {c["classe"] for c in a} == {"simples", "codigo", "complexo"}
    assert {c["idioma"] for c in a} == set(gp.IDIOMAS)
    assert len({c["chave"] for c in a}) == len(a)
    assert any(c["adversarial"] for c in a)
    assert all(c["linguagem"] in gp.LINGUAGENS for c in a if c["linguagem"])


def test_prompt_do_gerador_menciona_idioma_e_lote(gp):
    c = next(c for c in gp.montar_celulas(42) if c["linguagem"])
    texto = gp.montar_prompt_gerador(c)
    assert gp.IDIOMAS[c["idioma"]] in texto and c["linguagem"] in texto and str(c["lote"]) in texto


@pytest.mark.parametrize("bruto,limpo", [
    ("1. Qual a capital?", "Qual a capital?"),
    ("- item", "item"),
    ('"entre aspas"', "entre aspas"),
    ("  sem mudança  ", "sem mudança"),
])
def test_limpar(gp, bruto, limpo):
    assert gp.limpar(bruto) == limpo


def test_quase_duplicatas(gp):
    itens = [{"prompt": "Qual é a capital da França?"}, {"prompt": "Qual e a capital da Franca?!"},
             {"prompt": "Escreva uma função que soma dois números."}]
    manter, removidos = gp.remover_quase_duplicatas(itens, 0.9)
    assert removidos == 1 and len(manter) == 2


def _itens(n_por_grupo=10):
    out = []
    for classe in ("simples", "codigo", "complexo"):
        for idioma in ("pt", "en"):
            for k in range(n_por_grupo):
                out.append({"id": f"{classe}-{idioma}-{k}", "classe_esperada": classe, "subtipo": "x",
                            "idioma": idioma, "prompt": f"{classe} {idioma} {k}"})
    return out


def test_split_estratificado(gp):
    itens = _itens()
    gp.separar(itens, 0.3, seed=1)
    for classe in ("simples", "codigo", "complexo"):
        for idioma in ("pt", "en"):
            g = [i for i in itens if i["classe_esperada"] == classe and i["idioma"] == idioma]
            assert sum(i["split"] == "teste" for i in g) == 3


def test_subconjunto_balanceado(gp):
    itens = _itens()
    gp.separar(itens, 0.5, seed=1)
    sub = gp.subconjunto(itens, 6, seed=1)
    assert len(sub) == 18
    assert all(i["split"] == "teste" for i in sub)
    for classe in ("simples", "codigo", "complexo"):
        idiomas = {i["idioma"] for i in sub if i["classe_esperada"] == classe}
        assert idiomas == {"pt", "en"}  # alterna entre os grupos


def test_finalizar_ponta_a_ponta(gp, tmp_path, monkeypatch):
    bruto = tmp_path / "bruto.jsonl"
    celulas = gp.montar_celulas(42)
    linhas = []
    for classe in ("simples", "codigo", "complexo"):
        c = next(c for c in celulas if c["classe"] == classe)
        prompts = [f"Pedido {classe} número {k} sobre o tema {k * 7} com detalhes {k}" for k in range(8)]
        prompts += ["1. Qual é a capital da Austrália? Responda apenas com o nome da cidade.", "x"]
        linhas.append(json.dumps({"celula": c, "modelo": "falso", "prompts": prompts}, ensure_ascii=False))
    bruto.write_text("\n".join(linhas), encoding="utf-8")
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps([{"id": "S01", "classe_esperada": "simples",
                                   "prompt": "Qual é a capital da Austrália? Responda apenas com o nome da cidade."}]),
                      encoding="utf-8")
    for nome, valor in (("BRUTO", bruto), ("DESCARTADOS", tmp_path / "desc.json"), ("GOLDEN", golden),
                        ("SAIDA", tmp_path / "gerados.json"), ("SAIDA_EXPERIMENTO", tmp_path / "exp.json")):
        monkeypatch.setattr(gp, nome, valor)

    class A:
        seed, frac_teste, limiar_similaridade, subconjunto_por_classe, amostra_revisao = 42, 0.25, 0.97, 1, 0

    gp.finalizar(A)
    itens = json.loads((tmp_path / "gerados.json").read_text(encoding="utf-8"))
    # "x" é curto demais e a cópia do prompt manual é duplicata
    assert all(i["prompt"] != "x" for i in itens)
    assert not any("Austrália" in i["prompt"] for i in itens)
    assert {i["split"] for i in itens} == {"treino", "teste"}
    assert len({i["id"] for i in itens}) == len(itens)
    exp = json.loads((tmp_path / "exp.json").read_text(encoding="utf-8"))
    assert exp[0]["id"] == "S01" and len(exp) == 1 + 3


def test_gerar_refaz_lote_incompleto_e_guarda_o_melhor(gp, tmp_path, monkeypatch):
    bruto = tmp_path / "bruto.jsonl"
    monkeypatch.setattr(gp, "DADOS", tmp_path)
    monkeypatch.setattr(gp, "BRUTO", bruto)
    respostas = {1: 1, 2: 3, 3: 2}  # tentativa -> quantos prompts o "gerador" devolve
    chamadas = []

    def lote_falso(client, modelo, c, seed, temperatura):
        tentativa = seed - 42
        chamadas.append(tentativa)
        n = respostas[tentativa] if c["chave"] == "simples|factual|pt|0" else c["lote"]
        return [f"prompt {k} da célula {c['chave']}" for k in range(n)], {"eval_count": 1}

    monkeypatch.setattr(gp, "gerar_lote", lote_falso)

    class A:
        modelo, seed, limite, timeout, tentativas, temperatura = "falso", 42, 2, 5, 3, 0.9

    gp.gerar(A)
    linhas = [json.loads(l) for l in bruto.read_text(encoding="utf-8").splitlines()]
    assert chamadas == [1, 2, 3, 1]  # a 1ª célula esgota as tentativas; a 2ª sai na primeira
    primeira = next(l for l in linhas if l["celula"]["chave"] == "simples|factual|pt|0")
    assert len(primeira["prompts"]) == 3  # o melhor dos três lotes incompletos


# --------------------------------------------------------------------------- #
# rodar_experimento: modo carga
# --------------------------------------------------------------------------- #
def test_planejar_carga_alterna_cenarios_e_mantem_ordem(rx):
    prompts = [{"id": f"P{i}", "classe_esperada": "simples", "prompt": "x"} for i in range(10)]
    r1, r2 = rx.planejar_carga(prompts, 1, 42), rx.planejar_carga(prompts, 2, 42)
    assert [c for _, c, _ in r1] == ["baseline", "proposta"]
    assert [c for _, c, _ in r2] == ["proposta", "baseline"]
    assert [i["id"] for i in r1[0][2]] == [i["id"] for i in r1[1][2]]


def test_rodar_bloco_executa_em_paralelo(rx, monkeypatch):
    ativos, pico, trava = [0], [0], threading.Lock()

    def medir_falso(cenario, client, item):
        with trava:
            ativos[0] += 1
            pico[0] = max(pico[0], ativos[0])
        time.sleep(0.05)
        with trava:
            ativos[0] -= 1
        return {"cenario": cenario, "prompt_id": item["id"], "latencia_s": 0.05, "sucesso": True, "erro": ""}

    monkeypatch.setattr(rx, "medir", medir_falso)
    itens = [{"id": f"P{i}"} for i in range(8)]
    vistos = []
    dur = rx.rodar_bloco(None, itens, "baseline", 4, lambda k, item, med: vistos.append(k))
    assert sorted(vistos) == list(range(1, 9))
    assert pico[0] == 4
    assert dur < 8 * 0.05
