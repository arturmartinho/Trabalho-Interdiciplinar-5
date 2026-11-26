"""Fixtures comuns dos testes."""

import pytest

import src.data_access as data_access


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Evita que os testes usem ou criem o banco local do desenvolvedor."""
    monkeypatch.setattr(data_access, "DB_FILE", str(tmp_path / "test.db"))


@pytest.fixture(autouse=True)
def _classificador_deterministico(monkeypatch):
    """Testes não dependem do .env local nem de rede: regras + estimativa de tokens.

    Testes que exercitam o modelo treinado ou o tokenizador real passam a
    estratégia/tokenizador explicitamente.
    """
    import src.classificador as classificador
    import src.texto as texto

    monkeypatch.setattr(classificador, "ESTRATEGIA", "regras")
    monkeypatch.setattr(texto, "_tokenizador_nome", texto.ESTIMATIVA)
