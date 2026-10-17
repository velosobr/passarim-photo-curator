"""Testes da conferência dos pesos, sem baixar nada: criamos arquivos falsos
numa pasta temporária e conferimos que um hash errado é recusado."""

import hashlib

import pytest

from photo_curator import weights


def test_files_are_pinned():
    for f in weights.FILES:
        assert len(f.revision) == 40  # hash de commit do Hugging Face, nunca "main"
        assert len(f.sha256) == 64


def test_verify_rejects_wrong_hash(tmp_path, monkeypatch):
    fake = weights.WeightFile("x/y", "0" * 40, "a.bin", "sub", hashlib.sha256(b"certo").hexdigest())
    monkeypatch.setattr(weights, "FILES", (fake,))
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.bin").write_bytes(b"errado")
    with pytest.raises(RuntimeError, match="sha256"):
        weights.verify(tmp_path)
    (tmp_path / "sub" / "a.bin").write_bytes(b"certo")
    weights.verify(tmp_path)  # não levanta


def test_verify_rejects_missing_file(tmp_path, monkeypatch):
    fake = weights.WeightFile("x/y", "0" * 40, "a.bin", "sub", "0" * 64)
    monkeypatch.setattr(weights, "FILES", (fake,))
    with pytest.raises(RuntimeError, match="falta"):
        weights.verify(tmp_path)
