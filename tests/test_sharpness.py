"""Testes da medida de nitidez. Usamos imagens sintéticas: um tabuleiro de
xadrez (bordas fortes, muito nítido) e um cinza liso (nenhuma borda)."""

import numpy as np
from PIL import Image, ImageFilter

from photo_curator.sharpness import measure
from photo_curator.types import Box


def checkerboard(size: int = 200, cell: int = 10) -> Image.Image:
    yy, xx = np.indices((size, size))
    board = (((xx // cell) + (yy // cell)) % 2 * 255).astype(np.uint8)
    return Image.fromarray(board).convert("RGB")


def test_sharp_image_scores_higher_than_blurred():
    sharp = checkerboard()
    blurred = sharp.filter(ImageFilter.GaussianBlur(radius=4))
    assert measure(sharp, None) > measure(blurred, None)


def test_flat_image_is_zero_and_values_stay_in_range():
    flat = Image.new("RGB", (100, 100), (128, 128, 128))
    assert measure(flat, None) == 0.0
    assert 0.0 <= measure(checkerboard(), None) <= 1.0


def test_measures_only_inside_the_box():
    # Metade de cima nítida, metade de baixo lisa: a caixa na metade lisa dá 0.
    img = checkerboard(200)
    img.paste((128, 128, 128), (0, 100, 200, 200))
    assert measure(img, Box(0, 120, 200, 200, 0.9)) == 0.0
    assert measure(img, Box(0, 0, 200, 90, 0.9)) > 0.5


def test_sharpness_box_outside_image():
    # Caixa fora da foto (o detector às vezes erra): não pode quebrar.
    assert 0.0 <= measure(checkerboard(), Box(-50, -50, 900, 900, 0.9)) <= 1.0


def test_tiny_box_falls_back_to_whole_image():
    # Uma caixa de 1 px não tem vizinhos para o Laplaciano: usa a foto inteira.
    assert measure(checkerboard(), Box(10, 10, 11, 11, 0.9)) == measure(checkerboard(), None)
