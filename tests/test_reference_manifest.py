"""Testes do manifesto do conjunto de referência (CSV com uma linha por foto)."""

import pytest

from photo_curator.reference import manifest
from photo_curator.reference.manifest import Row


def row(photo_id: str, label: str = "SINGLE_BIRD") -> Row:
    return Row(
        photo_id,
        "9" + photo_id,
        "Crotophaga ani",
        "SINGLE_BIRD",
        label,
        "Fulana",
        "cc-by",
        "https://x/1",
        "species",
    )


def test_write_then_read_roundtrip_sorted_by_photo_id(tmp_path):
    path = tmp_path / "manifest.csv"
    manifest.write(path, [row("20"), row("3")])
    assert [r.photo_id for r in manifest.read(path)] == ["3", "20"]  # ordem numérica, não de texto
    assert manifest.read(path)[0] == row("3")


def test_read_rejects_wrong_columns(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text("photo_id,label\n1,SINGLE_BIRD\n", encoding="utf-8")
    with pytest.raises(ValueError, match="colunas"):
        manifest.read(path)


def test_attribution_lists_every_photo():
    text = manifest.attribution_markdown([row("1"), row("2")])
    assert "mantêm a licença" in text
    assert text.count("| Fulana |") == 2
    assert "https://x/1" in text


def test_image_path(tmp_path):
    assert manifest.image_path(tmp_path, row("7")) == tmp_path / "photos" / "7.webp"
