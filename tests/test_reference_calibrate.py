"""Testes da busca dos limites com características sintéticas."""

import pytest

from photo_curator import config
from photo_curator.reference import calibrate as cal
from photo_curator.reference.features import Features
from photo_curator.types import CLIP_CLASSES, Box


def feats(boxes=(), **clip: float) -> Features:
    scores = dict.fromkeys(CLIP_CLASSES, 0.0)
    scores.update(clip)
    return Features(boxes=tuple(boxes), clip=scores, sharpness=0.5, width=800, height=600)


BIRD = [Box(200, 150, 600, 450, 0.95)]
# 20 aves sozinhas claras e 20 fotos de ovos em que o CLIP dá só 0,4 para "eggs"
# (abaixo do limite padrão de 0,5): com o padrão, os ovos viram ave sozinha.
SAMPLES = [("SINGLE_BIRD", feats(BIRD, single=1.0))] * 20 + [("EGGS", feats(BIRD, eggs=0.4, single=0.6))] * 20


def test_default_thresholds_miss_the_eggs():
    report = cal.report_for(SAMPLES, config.DEFAULT_THRESHOLDS)
    assert report.discard_recall.value == 0.0


def test_search_lowers_the_eggs_threshold_just_enough():
    best, report = cal.search(SAMPLES)
    assert report.failures() == []
    # 0,4 é o valor da grade mais perto do padrão (0,5) que pega os ovos.
    assert best.discard_min["eggs"] == pytest.approx(0.4)
    # Os outros limites não tinham motivo para mudar.
    assert best.flock_min == config.FLOCK_MIN


def test_with_param_changes_one_discard_class():
    t = cal.with_param(config.DEFAULT_THRESHOLDS, "discard_min.nest", 0.3)
    assert t.discard_min["nest"] == 0.3
    assert t.discard_min["eggs"] == config.DISCARD_MIN["eggs"]
    assert config.DEFAULT_THRESHOLDS.discard_min["nest"] == config.DISCARD_MIN["nest"]  # o padrão não mudou


def test_config_snippet_is_valid_python():
    namespace: dict = {}
    # exec roda o texto como código Python: prova que o trecho cola sem erro no config.py.
    exec(cal.config_snippet(config.DEFAULT_THRESHOLDS), namespace)
    assert namespace["FLOCK_MIN"] == config.FLOCK_MIN
    assert namespace["DISCARD_MIN"] == config.DISCARD_MIN
