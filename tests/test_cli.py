"""Testes da CLI unificada e da ligação com o catálogo de modelos."""

import src.data_access as data_access
from src.ai_model import AIModelDM
from src.cli import build_parser
from src.loadbalancer import registry


def _use_temp_db(monkeypatch, tmp_path):
    monkeypatch.setattr(data_access, "DB_FILE", str(tmp_path / "test.db"))


def test_parser_expoe_os_comandos_principais():
    parser = build_parser()
    assert parser.parse_args(["models", "list"]).handler.__name__ == "_cmd_models_list"
    assert parser.parse_args(["serve"]).handler.__name__ == "_cmd_serve"
    assert parser.parse_args(["infer", "--prompt", "oi"]).handler.__name__ == "_cmd_infer"


def test_register_e_list_models(monkeypatch, tmp_path, capsys):
    _use_temp_db(monkeypatch, tmp_path)

    assert AIModelDM.register(
        "phi3.5",
        type_="local",
        role="complexo",
        status="up",
    )
    assert AIModelDM.register(
        "qwen2.5-coder:0.5b",
        type_="remote",
        url="http://host:11434",
        role="codigo",
        status="up",
    )

    rows = AIModelDM.list()
    assert {row["name"] for row in rows} == {"phi3.5", "qwen2.5-coder:0.5b"}
    assert {row["role"] for row in rows} == {"complexo", "codigo"}

    parser = build_parser()
    args = parser.parse_args(["models", "list"])
    assert args.handler(args) == 0
    saida = capsys.readouterr().out
    assert "phi3.5" in saida
    assert "complexo" in saida


def test_loadbalancer_le_as_classes_do_catalogo(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)

    AIModelDM.register("meu-simples", role="simples", status="up")
    AIModelDM.register("meu-codigo", role="codigo", status="up")
    AIModelDM.register(
        "meu-complexo",
        type_="remote",
        url="http://remoto:11434",
        role="complexo",
        status="up",
    )
    AIModelDM.register("meu-reserva", role="reserva", status="up")

    assert registry.model_roles() == {
        "simples": "meu-simples",
        "codigo": "meu-codigo",
        "complexo": "meu-complexo",
    }
    assert registry.model_urls()["meu-complexo"] == "http://remoto:11434"
    assert registry.fallback_model() == "meu-reserva"
