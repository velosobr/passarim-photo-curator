"""Reduz uma foto baixada para guardar no conjunto de referência.

512 px no lado maior bastam: o CLIP olha a foto em 224 px e o detector em
640 px. Fotos menores deixam o repositório leve (~30 KB cada).

exif_transpose: câmeras guardam a foto "deitada" e anotam no EXIF (metadados
da imagem) para girar na hora de mostrar. Aplicamos a rotação antes de reduzir,
para a foto ficar de pé como a pessoa a vê.

Quem usa: scripts/build_reference.py.
"""

import io

from PIL import Image, ImageOps

MAX_SIDE = 512


def shrink(data: bytes) -> bytes:
    """Bytes de uma imagem qualquer -> WebP com no máximo MAX_SIDE no lado maior."""
    with Image.open(io.BytesIO(data)) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
    # thumbnail reduz mantendo a proporção e nunca amplia.
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    out = io.BytesIO()
    image.save(out, format="WEBP", quality=80)  # a mesma qualidade do worker
    return out.getvalue()
