"""Testes da leitura da configuração do ponto de entrada (sem subir modelos)."""

from pathlib import Path

from photo_curator.__main__ import Settings


def test_defaults():
    s = Settings.from_env({})
    assert s.port == 50061
    assert s.models_dir == Path("/opt/models")


def test_env_overrides():
    s = Settings.from_env({"PORT": "6000", "MODELS_DIR": "/tmp/m"})
    assert s.port == 6000
    assert s.models_dir == Path("/tmp/m")
