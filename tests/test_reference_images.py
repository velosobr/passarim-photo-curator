"""Testes da redução das fotos para o conjunto de referência."""

import io

from PIL import Image

from photo_curator.reference.images import MAX_SIDE, shrink


def jpeg(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (10, 120, 30)).save(buf, format="JPEG")
    return buf.getvalue()


def test_landscape_is_reduced_to_512_wide():
    out = Image.open(io.BytesIO(shrink(jpeg(1024, 600))))
    assert out.format == "WEBP"
    assert out.size == (MAX_SIDE, 300)


def test_portrait_is_reduced_to_512_tall():
    assert Image.open(io.BytesIO(shrink(jpeg(600, 1024)))).size == (300, MAX_SIDE)


def test_small_image_is_not_enlarged():
    assert Image.open(io.BytesIO(shrink(jpeg(200, 100)))).size == (200, 100)
