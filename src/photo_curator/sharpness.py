"""Nitidez da foto (ou da maior ave), de 0 a 1.

Como medimos: aplicamos o Laplaciano, uma conta que olha cada pixel e os 4
vizinhos (cima, baixo, esquerda, direita) e dá um número alto onde há borda
forte e perto de zero onde a imagem é lisa. Numa foto nítida, as bordas são
fortes e esse número varia muito de pixel para pixel; numa foto borrada, quase
não varia. A "variância do Laplaciano" mede justamente essa variação.

Como a variância vai de ~10 (borrada) a ~10 000 (muito nítida), usamos log10 e
levamos o intervalo [SHARPNESS_LOG_MIN, SHARPNESS_LOG_MAX] do config para 0..1.

Quem usa: server.py, uma vez por foto, no recorte da maior ave.
"""

import math

import numpy as np
from PIL import Image

from photo_curator import config
from photo_curator.rules import clamp_box
from photo_curator.types import Box

# Abaixo disso, a caixa não tem pixels vizinhos suficientes: usa a foto inteira.
_MIN_CROP_SIDE = 3


def _laplacian_variance(gray: np.ndarray) -> float:
    """Variância do Laplaciano de uma imagem em tons de cinza (array 2D)."""
    g = gray.astype(np.float64)
    # Laplaciano com o "kernel" de 4 vizinhos: 4*centro - cima - baixo - esq - dir.
    # Fatiar o array evita laços em Python (numpy faz a conta toda de uma vez).
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def measure(image: Image.Image, box: Box | None) -> float:
    """Nitidez de 0 a 1 do recorte da caixa (ou da foto inteira, se box é None)."""
    crop = image
    if box is not None:
        b = clamp_box(box, image.width, image.height)
        if b.x1 - b.x0 >= _MIN_CROP_SIDE and b.y1 - b.y0 >= _MIN_CROP_SIDE:
            crop = image.crop((int(b.x0), int(b.y0), int(b.x1), int(b.y1)))
    gray = np.asarray(crop.convert("L"))
    if gray.shape[0] < _MIN_CROP_SIDE or gray.shape[1] < _MIN_CROP_SIDE:
        return 0.0
    variance = _laplacian_variance(gray)
    if variance <= 0:
        return 0.0
    span = config.SHARPNESS_LOG_MAX - config.SHARPNESS_LOG_MIN
    normalized = (math.log10(variance) - config.SHARPNESS_LOG_MIN) / span
    return min(max(normalized, 0.0), 1.0)
