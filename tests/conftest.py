"""Fixtures comuns dos testes."""

import pytest

import src.data_access as data_access


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Evita que os testes usem ou criem o banco local do desenvolvedor."""
    monkeypatch.setattr(data_access, "DB_FILE", str(tmp_path / "test.db"))
