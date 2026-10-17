"""Testes da aplicação da revisão do dono ao manifesto."""

import pytest

from photo_curator.reference.manifest import EXCLUDED, Row
from photo_curator.reference.review import apply, missing


def row(photo_id: str, label: str = "") -> Row:
    return Row(photo_id, "1", "sp", "NEST", label, "a", "cc-by", "u", "nest")


def test_apply_fills_labels():
    rows = apply([row("1"), row("2")], {"1": "SINGLE_BIRD", "2": EXCLUDED})
    assert [r.label for r in rows] == ["SINGLE_BIRD", EXCLUDED]


def test_apply_keeps_labels_not_in_review():
    # Uma revisão nova (ex.: só das fotos acrescentadas) não apaga as antigas.
    assert apply([row("1", "EGGS"), row("2")], {"2": "NEST"})[0].label == "EGGS"


def test_apply_rejects_unknown_photo():
    with pytest.raises(ValueError, match="99"):
        apply([row("1")], {"99": "NEST"})


def test_apply_rejects_unknown_label():
    with pytest.raises(ValueError, match="PASSARO"):
        apply([row("1")], {"1": "PASSARO"})


def test_missing_lists_unlabeled():
    assert missing([row("1", "NEST"), row("2"), row("3")]) == ["2", "3"]
