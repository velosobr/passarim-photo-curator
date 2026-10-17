"""Testes da agregação das frases do CLIP em classes."""

import pytest

from photo_curator.prompts import PROMPTS, aggregate, flat_prompts
from photo_curator.types import CLIP_CLASSES


def test_every_clip_class_has_prompts():
    assert tuple(PROMPTS) == CLIP_CLASSES
    assert all(len(texts) >= 1 for texts in PROMPTS.values())


def test_flat_prompts_follow_class_order():
    expected = [t for name in CLIP_CLASSES for t in PROMPTS[name]]
    assert flat_prompts() == expected


def test_aggregate_sums_probabilities_per_class():
    n = len(flat_prompts())
    probs = [0.0] * n
    # Toda a probabilidade nas frases de "eggs".
    start = sum(len(PROMPTS[c]) for c in CLIP_CLASSES[: CLIP_CLASSES.index("eggs")])
    for i in range(len(PROMPTS["eggs"])):
        probs[start + i] = 1.0 / len(PROMPTS["eggs"])
    scores = aggregate(probs)
    assert scores["eggs"] == pytest.approx(1.0)
    assert sum(scores.values()) == pytest.approx(1.0)
    assert set(scores) == set(CLIP_CLASSES)


def test_aggregate_rejects_wrong_length():
    with pytest.raises(ValueError):
        aggregate([1.0])
