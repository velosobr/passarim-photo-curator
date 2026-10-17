"""Testes com os modelos de VERDADE nas três fotos reais que motivaram a frente A.

Marcados com "models": não rodam no `pytest` comum (ver addopts no
pyproject.toml). Rodam com `pytest -m models` e precisam da variável
CURATOR_MODELS_DIR apontando para a pasta criada por
`python -m photo_curator.weights <pasta>`.
"""

import io
import os
import time
from pathlib import Path

import pytest
from PIL import Image

from photo_curator import config, weights
from photo_curator.detector import RtDetrDetector
from photo_curator.rules import curate, largest_box
from photo_curator.sharpness import measure
from photo_curator.types import Category
from photo_curator.zeroshot import ClipZeroShot

pytestmark = pytest.mark.models
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def models():
    root = os.environ.get("CURATOR_MODELS_DIR")
    if not root:
        pytest.skip("defina CURATOR_MODELS_DIR (pasta dos pesos)")
    root = Path(root)
    weights.verify(root)
    return RtDetrDetector(root / weights.RTDETR_SUBDIR), ClipZeroShot(root / weights.CLIP_FILE)


def classify(models, name):
    detector, zeroshot = models
    image = Image.open(FIXTURES / name).convert("RGB")
    boxes = detector.detect(image)
    return curate(
        boxes, zeroshot.classify(image), measure(image, largest_box(boxes)), image.width, image.height
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("andorinha-bando.webp", Category.FLOCK),
        ("anu-ovos.webp", Category.EGGS),
        ("arara-sozinha.webp", Category.SINGLE_BIRD),
    ],
)
def test_real_photos(models, name, expected):
    assert classify(models, name).category is expected


def test_detector_only_returns_confident_birds(models):
    detector, _ = models
    boxes = detector.detect(Image.open(FIXTURES / "arara-sozinha.webp").convert("RGB"))
    assert len(boxes) == 1
    assert all(b.score >= config.DETECTOR_MIN_SCORE for b in boxes)


def test_clip_probabilities_sum_to_one(models):
    _, zeroshot = models
    scores = zeroshot.classify(Image.open(FIXTURES / "anu-ovos.webp").convert("RGB"))
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-4)


def test_batch_of_eight_is_fast_enough(models):
    # 8 fotos (as 3 reais, repetidas) em até 15 s na máquina da CI.
    names = ["andorinha-bando.webp", "anu-ovos.webp", "arara-sozinha.webp"] * 3
    start = time.perf_counter()
    for name in names[:8]:
        classify(models, name)
    assert time.perf_counter() - start < 15.0


def test_webp_bytes_roundtrip(models):
    # Mesmo caminho do servidor: bytes -> Image. Garante que o Pillow lê os WebP do worker.
    data = (FIXTURES / "arara-sozinha.webp").read_bytes()
    assert Image.open(io.BytesIO(data)).format == "WEBP"
