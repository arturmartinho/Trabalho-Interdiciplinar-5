import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from classificador import (  # noqa: E402
    CLASSE_CODIGO,
    CLASSE_COMPLEXO,
    CLASSE_SIMPLES,
    LIMITE_LONGO,
    MODELO_CODIGO,
    MODELO_COMPLEXO,
    MODELO_SIMPLES,
    classificar,
    classificar_e_registrar,
)


# --------------------------------------------------------------------------- #
# Os três casos principais
# --------------------------------------------------------------------------- #
def test_pedido_curto_factual_vai_para_ailo():
    r = classificar("Qual é a capital da França?")
    assert r.classe == CLASSE_SIMPLES
    assert r.modelo == MODELO_SIMPLES


def test_pedido_de_codigo_vai_para_qwen():
    r = classificar("Escreva uma função em Python que inverte uma string.")
    assert r.classe == CLASSE_CODIGO
    assert r.modelo == MODELO_CODIGO


def test_pedido_de_raciocinio_vai_para_phi():
    r = classificar("Explique passo a passo por que o céu é azul.")
    assert r.classe == CLASSE_COMPLEXO
    assert r.modelo == MODELO_COMPLEXO


def test_pedido_longo_vai_para_phi():
    prompt = "palavra " * (LIMITE_LONGO + 1)
    r = classificar(prompt)
    assert r.classe == CLASSE_COMPLEXO
    assert "longo" in r.motivo


# --------------------------------------------------------------------------- #
# Casos ambíguos
# --------------------------------------------------------------------------- #
def test_ambiguo_codigo_e_raciocinio_prioriza_codigo():
    r = classificar("Explique por que esse código Python fica lento com listas grandes.")
    assert r.classe == CLASSE_CODIGO
    assert r.modelo == MODELO_CODIGO
    assert "prioridade" in r.motivo


def test_ambiguo_tamanho_medio_sem_sinais_vai_para_phi():
    prompt = (
        "Estou organizando uma viagem com a família para o litoral no fim do ano "
        "e gostaria de sugestões de lugares tranquilos, com boa comida e que não "
        "sejam muito caros para ficar uma semana"
    )
    r = classificar(prompt)
    assert r.classe == CLASSE_COMPLEXO
    assert "ambiguo" in r.motivo


# --------------------------------------------------------------------------- #
# Propriedades da função
# --------------------------------------------------------------------------- #
def test_e_deterministica():
    prompt = "Implemente uma API em FastAPI"
    assert classificar(prompt) == classificar(prompt)


def test_ignora_acentos_e_maiusculas():
    assert classificar("FUNÇÃO de soma").classe == CLASSE_CODIGO
    assert classificar("funcao de soma").classe == CLASSE_CODIGO


def test_prompt_vazio_nao_quebra():
    r = classificar("")
    assert r.classe == CLASSE_SIMPLES


def test_registra_no_log(capsys):
    r = classificar_e_registrar("Qual é a capital da França?", prompt_id="s01")
    saida = capsys.readouterr().out
    assert "id=s01" in saida
    assert f"classe={r.classe}" in saida
    assert f"modelo={r.modelo}" in saida
    assert "motivo=" in saida
