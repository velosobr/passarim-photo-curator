"""Testes da sugestão do limite de baixa confiança (LOW_CONFIDENCE do worker)."""

import pytest

from photo_curator.reference.confidence import suggest_low_confidence


def test_suggestion_is_the_20th_percentile():
    # 10 valores: o 2º menor deixa 20% das fotos abaixo ou no limite.
    values = [0.9, 0.1, 0.5, 0.3, 0.7, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert suggest_low_confidence(values) == 0.2


def test_suggestion_with_one_value_is_that_value():
    assert suggest_low_confidence([0.42]) == 0.42


def test_suggestion_is_rounded_to_three_places():
    assert suggest_low_confidence([0.123456]) == 0.123


def test_suggestion_without_values_fails():
    with pytest.raises(ValueError, match="SINGLE_BIRD"):
        suggest_low_confidence([])
