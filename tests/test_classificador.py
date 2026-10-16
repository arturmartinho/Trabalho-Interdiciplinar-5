import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from avaliar_classificador import CSV_PADRAO, avaliar, carregar  # noqa: E402
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
    detectar_nivel_bloom,
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
        "Estou organizando uma viagem com a família para o litoral no fim do ano, "
        "tenho uma semana de férias, um orçamento apertado e duas crianças pequenas "
        "que enjoam fácil em estrada longa"
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
    assert f"bloom={r.nivel_bloom}" in saida


# --------------------------------------------------------------------------- #
# Taxonomia de Bloom
# --------------------------------------------------------------------------- #
def test_detecta_cada_nivel_de_bloom():
    assert detectar_nivel_bloom("Quem pintou a Mona Lisa?") == (1, "Lembrar")
    assert detectar_nivel_bloom("Descreva o ciclo da água.") == (2, "Entender")
    assert detectar_nivel_bloom("Calcule a área de um círculo.") == (3, "Aplicar")
    assert detectar_nivel_bloom("Compare vírus e bactérias.") == (4, "Analisar")
    assert detectar_nivel_bloom("Avalie essa decisão.") == (5, "Avaliar")
    assert detectar_nivel_bloom("Crie um plano de estudos.") == (6, "Criar")


def test_sem_verbo_e_nivel_lembrar():
    assert detectar_nivel_bloom("Bom dia, tudo bem?") == (1, "Lembrar")


def test_usa_o_maior_nivel_encontrado():
    nivel, _ = detectar_nivel_bloom("Explique e depois avalie as duas propostas.")
    assert nivel == 5


def test_lembrar_curto_vai_para_ailo():
    r = classificar("Liste três países da América do Sul.")
    assert r.classe == CLASSE_SIMPLES
    assert r.nivel_bloom == 1


def test_criar_fora_de_codigo_vai_para_phi():
    r = classificar("Escreva um poema sobre o mar.")
    assert r.classe == CLASSE_COMPLEXO
    assert r.nivel_bloom == 6
    assert "Criar" in r.motivo


def test_pergunta_curta_de_analise_vai_para_phi():
    r = classificar("Qual a diferença entre vírus e bactéria?")
    assert r.classe == CLASSE_COMPLEXO
    assert r.nivel_bloom == 4


def test_pedido_de_sugestoes_e_criar():
    prompt = (
        "Estou organizando uma viagem com a família para o litoral no fim do ano "
        "e gostaria de sugestões de lugares tranquilos, com boa comida e que não "
        "sejam muito caros para ficar uma semana"
    )
    r = classificar(prompt)
    assert r.classe == CLASSE_COMPLEXO
    assert r.nivel_bloom == 6


def test_preterito_nao_conta_como_verbo_de_comando():
    r = classificar("Quem escreveu Dom Casmurro?")
    assert r.classe == CLASSE_SIMPLES
    assert r.nivel_bloom == 1


def test_codigo_tem_prioridade_sobre_bloom():
    r = classificar("Escreva uma função em Python que inverte uma string.")
    assert r.classe == CLASSE_CODIGO
    assert r.nivel_bloom == 6
    assert "prioridade" in r.motivo


# --------------------------------------------------------------------------- #
# Acurácia no conjunto rotulado
# --------------------------------------------------------------------------- #
def test_acuracia_no_conjunto_rotulado():
    av = avaliar(carregar(CSV_PADRAO))
    assert av.total >= 70
    assert av.acuracia >= 0.90


def test_acuracia_no_conjunto_do_experimento():
    av = avaliar(carregar(CSV_PADRAO.parent.parent / "src" / "resources" / "prompts.json"))
    assert av.total >= 40
    assert av.acuracia >= 0.90
