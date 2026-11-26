"""Testes de tokens, do classificador treinado e do modo híbrido (sem rede)."""

import asyncio
import json

import pytest

import src.classificador as cl
import src.texto as texto

sklearn = pytest.importorskip("sklearn")

from src.classificador_ml import ModeloML, carregar, construir_pipeline  # noqa: E402


# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #
def test_estimativa_de_tokens():
    assert texto.estimar_tokens("") == 0
    assert texto.estimar_tokens("   ") == 0
    assert texto.estimar_tokens("a") == 1
    curto, longo = texto.estimar_tokens("abc " * 10), texto.estimar_tokens("abc " * 100)
    assert longo > curto * 9


def test_tokenizador_de_arquivo(tmp_path, monkeypatch):
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    tok = tokenizers.Tokenizer(WordLevel({"[UNK]": 0, "ola": 1}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    caminho = tmp_path / "tokenizer.json"
    tok.save(str(caminho))
    monkeypatch.setattr(texto, "_tokenizador_nome", str(caminho))
    assert texto.contar_tokens("ola mundo , tudo bem") == 5
    assert texto.tokenizador_ativo() == str(caminho)


def test_tokenizador_invalido_cai_para_estimativa(tmp_path, monkeypatch):
    ruim = tmp_path / "ruim.json"
    ruim.write_text("{nao e json", encoding="utf-8")
    monkeypatch.setattr(texto, "_tokenizador_nome", str(ruim))
    assert texto.contar_tokens("qualquer coisa") == texto.estimar_tokens("qualquer coisa")
    assert texto.tokenizador_ativo() == texto.ESTIMATIVA


def test_regras_em_palavras_reproduz_classificador_antigo():
    prompt = "palavra " * 130  # 130 palavras > 120
    classe, motivo, n = cl.classificar_regras(prompt, unidade="palavras")
    assert (classe, n) == (cl.CLASSE_COMPLEXO, 130)
    assert "palavras" in motivo


def test_resultado_traz_tokens_e_estrategia():
    r = cl.classificar("Qual é a capital da França?")
    assert r.tokens > 0 and r.palavras == 6
    assert r.estrategia == "regras" and r.confianca is None


# --------------------------------------------------------------------------- #
# Modelo treinado (TF-IDF)
# --------------------------------------------------------------------------- #
TREINO = {
    "simples": ["Qual a capital do Peru?", "Quanto é 7 vezes 8?", "Traduza 'gato' para inglês.",
                "Qual o plural de pão?", "Quem pintou a Mona Lisa?", "What is the capital of Spain?",
                "Oi, tudo bem?", "Quantos dias tem um ano bissexto?"],
    "codigo": ["Escreva uma função em Python que soma uma lista.", "Corrija este erro: TypeError em for x in None",
               "Como ordeno uma lista de dicionários pela chave idade?", "Write a JavaScript function to reverse a string.",
               "Crie uma consulta SQL que conte pedidos por cliente.", "Faça testes unitários para a função soma.",
               "Como leio um arquivo linha por linha em Go?", "Refatore este loop para usar map."],
    "complexo": ["Prove que existem infinitos números primos.", "Compare democracia direta e representativa.",
                 "Analise as causas da Primeira Guerra Mundial.", "Discuta os dilemas éticos da clonagem humana.",
                 "Planeje uma viagem de 10 dias com orçamento de 5 mil reais e três cidades.",
                 "Explain step by step why the sky is blue.", "Demonstre que raiz de 2 é irracional.",
                 "Avalie os prós e contras da energia nuclear para o Brasil."],
}


@pytest.fixture(scope="module")
def caminho_modelo(tmp_path_factory):
    X = [p for ps in TREINO.values() for p in ps]
    y = [c for c, ps in TREINO.items() for _ in ps]
    pipe = construir_pipeline("tfidf", C=8.0)
    pipe.fit(X, y)
    caminho = tmp_path_factory.mktemp("modelos") / "classificador_tfidf.joblib"
    ModeloML(pipe, "tfidf", {"n_treino": len(X)}).salvar(caminho)
    return str(caminho)


def test_modelo_salvo_e_recarregado(caminho_modelo):
    m = carregar(caminho_modelo)
    assert set(m.classes) == {"simples", "codigo", "complexo"}
    classe, conf, probs = m.prever("Qual a capital do Chile?")
    assert classe == "simples"
    assert 0 < conf <= 1 and abs(sum(probs.values()) - 1) < 1e-3


def test_estrategia_ml(caminho_modelo):
    r = cl.classificar("Prove que existem infinitos primos gêmeos?", estrategia="ml", caminho_modelo=caminho_modelo)
    assert r.estrategia == "ml"
    assert r.classe == "complexo"
    assert r.confianca is not None and r.motivo.startswith("ml[tfidf]")


def test_ml_e_deterministico(caminho_modelo):
    p = "Como ordeno uma lista pela idade?"
    assert (cl.classificar(p, estrategia="ml", caminho_modelo=caminho_modelo)
            == cl.classificar(p, estrategia="ml", caminho_modelo=caminho_modelo))


def test_hibrido_sintaxe_de_codigo_decide_sem_modelo(tmp_path):
    prompt = "o que isso faz?\n```\nfor i in range(3):\n    print(i)\n```"
    r = cl.classificar(prompt, estrategia="hibrido", caminho_modelo=str(tmp_path / "nao_existe.joblib"))
    assert r.classe == "codigo"
    assert "sintaxe de codigo" in r.motivo


def test_sem_modelo_usa_regras(tmp_path):
    r = cl.classificar("Explique por que o céu é azul.", estrategia="hibrido",
                       caminho_modelo=str(tmp_path / "nao_existe.joblib"))
    assert r.classe == "complexo"
    assert "indisponivel" in r.motivo


@pytest.mark.parametrize("prompt", [
    "From Paris to Rome, which city has more museums?",
    "What is the role of public health agencies?",
    "Qual é a capital da França?",
])
def test_sinais_fortes_nao_disparam_em_texto_comum(prompt):
    assert not cl._sinais_encontrados(cl.normalizar(prompt), cl._RE_CODIGO_FORTE)


@pytest.mark.parametrize("prompt", [
    "def soma(a, b):\n    return a + b",
    "import numpy as np\nx = np.zeros(3)",
    "const x = 10;",
    "SELECT nome FROM clientes WHERE idade > 30",
    "Traceback (most recent call last):\n  File \"a.py\", line 1",
])
def test_sinais_fortes_disparam_em_codigo(prompt):
    assert cl._sinais_encontrados(cl.normalizar(prompt), cl._RE_CODIGO_FORTE)


class _Falso:
    features = "falso"
    metadados = {}

    def __init__(self, classe, conf):
        self.classe, self.conf = classe, conf

    def prever(self, texto):
        return self.classe, self.conf, {self.classe: self.conf}


def test_hibrido_incerto_usa_regras(monkeypatch):
    monkeypatch.setattr(cl, "modelo_ml", lambda caminho=None: _Falso("simples", 0.40))
    r = cl.classificar("Explique por que o céu é azul.", estrategia="hibrido")
    assert r.classe == "complexo"  # regras: sinal de raciocínio
    assert "incerto" in r.motivo


def test_hibrido_confiante_usa_modelo(monkeypatch):
    monkeypatch.setattr(cl, "modelo_ml", lambda caminho=None: _Falso("codigo", 0.90))
    r = cl.classificar("Como ordeno uma lista pela idade?", estrategia="hibrido")
    assert r.classe == "codigo" and r.confianca == 0.90


@pytest.mark.parametrize("prompt", [
    "Prove que existem infinitos números primos.",
    "Explain step by step why the sky is blue.",
    "Explícame paso a paso por qué el cielo es azul.",
])
def test_hibrido_nao_rebaixa_raciocinio_explicito_para_simples(monkeypatch, prompt):
    monkeypatch.setattr(cl, "modelo_ml", lambda caminho=None: _Falso("simples", 0.95))
    r = cl.classificar(prompt, estrategia="hibrido")
    assert r.classe == "complexo"
    assert "nao rebaixa" in r.motivo


def test_hibrido_confiante_em_simples_sem_sinal_fica_simples(monkeypatch):
    monkeypatch.setattr(cl, "modelo_ml", lambda caminho=None: _Falso("simples", 0.95))
    assert cl.classificar("Who created Python?", estrategia="hibrido").classe == "simples"


def test_hibrido_nunca_manda_prompt_longo_para_simples(monkeypatch):
    monkeypatch.setattr(cl, "modelo_ml", lambda caminho=None: _Falso("simples", 0.95))
    r = cl.classificar("conte uma historia " * 200, estrategia="hibrido")
    assert r.classe == "complexo"
    assert "longo demais" in r.motivo


def test_estrategia_desconhecida_vira_regras():
    assert cl.classificar("oi", estrategia="xyz").estrategia == "regras"


# --------------------------------------------------------------------------- #
# Integração com o despacho
# --------------------------------------------------------------------------- #
def test_resolve_classifier_assincrono_quando_usa_rede(monkeypatch):
    from src.loadbalancer import dispatcher

    monkeypatch.setattr(cl, "usa_rede", lambda estrategia=None: True)
    monkeypatch.setattr(cl, "preparar", lambda estrategia=None, aquecer=True: {"estrategia": "teste"})
    f = dispatcher.resolve_classifier()
    assert asyncio.iscoroutinefunction(f)
    r = asyncio.run(f("Qual é a capital da França?", prompt_id="x"))
    assert r.classe == "simples"


def test_cli_classify_json(capsys):
    from src.cli import main

    assert main(["classify", "--prompt", "Qual é a capital da França?", "--estrategia", "regras", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["classe"] == "simples" and d["tokens"] > 0 and d["tokenizador"] == "estimativa"
