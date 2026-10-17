"""Testes das características em cache (com modelos falsos e imagens sintéticas)."""

import dataclasses

import pytest
from PIL import Image

from photo_curator import config
from photo_curator.reference import features as ft
from photo_curator.reference.manifest import Row
from photo_curator.types import CLIP_CLASSES, Box


def row(photo_id: str) -> Row:
    return Row(photo_id, "1", "sp", "SINGLE_BIRD", "SINGLE_BIRD", "a", "cc-by", "u", "species")


class CountingDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return [Box(10, 10, 100, 80, 0.9)]


class FakeZeroShot:
    def classify(self, image):
        scores = dict.fromkeys(CLIP_CLASSES, 0.0)
        scores["single"] = 1.0
        return scores


def write_photo(folder, photo_id):
    (folder / "photos").mkdir(exist_ok=True)
    Image.new("RGB", (200, 150), (50, 90, 40)).save(folder / "photos" / f"{photo_id}.webp", format="WEBP")


def test_compute_all_uses_cache(tmp_path):
    write_photo(tmp_path, "1")
    detector, cache = CountingDetector(), tmp_path / "cache.json"
    first = ft.compute_all([row("1")], tmp_path, detector, FakeZeroShot(), cache)
    second = ft.compute_all([row("1")], tmp_path, detector, FakeZeroShot(), cache)
    assert detector.calls == 1  # a segunda vez veio do cache
    assert first == second
    assert ft.predict(first["1"]) == "SINGLE_BIRD"
    assert first["1"].width == 200


def test_cache_with_other_key_is_ignored(tmp_path, monkeypatch):
    write_photo(tmp_path, "1")
    detector, cache = CountingDetector(), tmp_path / "cache.json"
    ft.compute_all([row("1")], tmp_path, detector, FakeZeroShot(), cache)
    monkeypatch.setattr(ft, "cache_key", lambda: "outra-versão-dos-modelos")
    ft.compute_all([row("1")], tmp_path, detector, FakeZeroShot(), cache)
    assert detector.calls == 2


def test_compute_all_fails_on_missing_image(tmp_path):
    with pytest.raises(FileNotFoundError, match="42"):
        ft.compute_all([row("42")], tmp_path, CountingDetector(), FakeZeroShot(), tmp_path / "c.json")


def test_predict_uses_given_thresholds():
    f = ft.Features(
        boxes=(),
        clip={**dict.fromkeys(CLIP_CLASSES, 0.0), "flock": 0.7},
        sharpness=0.5,
        width=100,
        height=100,
    )
    assert ft.predict(f) == "FLOCK"
    assert ft.predict(f, dataclasses.replace(config.DEFAULT_THRESHOLDS, flock_min=0.8)) == "OTHER"
