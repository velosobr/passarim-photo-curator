"""Testes das métricas. Os números esperados foram contados à mão no conjunto
abaixo, e os do intervalo de Wilson, conferidos com a fórmula."""

import pytest

from photo_curator.reference.metrics import evaluate, wilson

# 10 aves sozinhas: 8 acertos, 1 vista como bando, 1 como ninho.
# 2 bandos: 1 visto como ave sozinha, 1 acerto.
# 5 descartes: ninho certo, ninho visto como ave sozinha, ninho que não abriu,
# ovos certos, ovos vistos como bando.
PAIRS = (
    [("SINGLE_BIRD", "SINGLE_BIRD")] * 8
    + [("SINGLE_BIRD", "FLOCK"), ("SINGLE_BIRD", "NEST")]
    + [("FLOCK", "SINGLE_BIRD"), ("FLOCK", "FLOCK")]
    + [
        ("NEST", "NEST"),
        ("NEST", "SINGLE_BIRD"),
        ("NEST", "UNSPECIFIED"),
        ("EGGS", "EGGS"),
        ("EGGS", "FLOCK"),
    ]
)


def test_rates_counted_by_hand():
    r = evaluate(PAIRS)
    # Previstas como ave sozinha: 8 certas + 1 bando + 1 ninho = 10.
    assert (r.single_precision.hits, r.single_precision.total) == (8, 10)
    assert (r.single_recall.hits, r.single_recall.total) == (8, 10)
    # Descartes pegos: ninho→ninho, ninho→não abriu, ovos→ovos. Ovos→bando NÃO conta.
    assert (r.discard_recall.hits, r.discard_recall.total) == (3, 5)
    assert (r.accuracy.hits, r.accuracy.total) == (11, 17)
    assert r.confusion[("SINGLE_BIRD", "SINGLE_BIRD")] == 8
    assert r.confusion[("EGGS", "FLOCK")] == 1


def test_failures_and_deficit():
    r = evaluate(PAIRS)
    # Falham a precisão (80%) e os descartes (60%); a revocação (80%) passa dos 70%.
    assert len(r.failures()) == 2
    # (0,95 - 0,8) + (0,70 - 0,80 -> 0) + (0,90 - 0,6) = 0,45
    assert r.deficit() == pytest.approx(0.45)
    perfect = evaluate([("SINGLE_BIRD", "SINGLE_BIRD"), ("NEST", "NEST")])
    assert perfect.failures() == []
    assert perfect.deficit() == 0.0


def test_wilson_known_values():
    assert wilson(19, 20) == pytest.approx((0.7639, 0.9911), abs=1e-4)
    assert wilson(10, 10) == pytest.approx((0.7225, 1.0), abs=1e-4)
    assert wilson(0, 0) == (0.0, 1.0)  # sem dados: não sabemos nada


def test_empty_rate_fails_its_target():
    # Nenhuma foto de descarte: a meta não pode passar "por vazio".
    r = evaluate([("SINGLE_BIRD", "SINGLE_BIRD")])
    assert any("descartes" in f for f in r.failures())


def test_markdown_has_targets_and_confusion():
    text = evaluate(PAIRS).markdown()
    assert "Precisão de SINGLE_BIRD" in text
    assert "80.0%" in text
    assert "| SINGLE_BIRD |" in text  # linha da matriz de confusão
