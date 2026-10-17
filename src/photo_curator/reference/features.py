"""As "características" de cada foto: o que os modelos viram, guardado em cache.

Rodar o detector e o CLIP leva ~0,5 s por foto. A regra (rules.curate), ao
contrário, é instantânea. Então rodamos os modelos UMA vez por foto, guardamos
as saídas (caixas, probabilidades do CLIP, nitidez e tamanho) num arquivo
JSON, e a calibração experimenta milhares de combinações de limites só com a
regra.

O cache tem uma CHAVE: o hash dos pesos, das frases do CLIP e da confiança
mínima do detector. Se qualquer um mudar, as saídas mudam, e o cache antigo é
ignorado (senão a calibração usaria saídas de outro modelo).

Quem usa: reference/calibrate.py, scripts/calibrate.py e scripts/measure.py.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from photo_curator import config, weights
from photo_curator.config import DEFAULT_THRESHOLDS, Thresholds
from photo_curator.models import Detector, ZeroShot
from photo_curator.prompts import flat_prompts
from photo_curator.reference.manifest import Row, image_path
from photo_curator.rules import curate, largest_box
from photo_curator.sharpness import measure
from photo_curator.types import Box


@dataclass(frozen=True)
class Features:
    boxes: tuple[Box, ...]
    clip: dict[str, float]
    sharpness: float
    width: int
    height: int


def predict(f: Features, thresholds: Thresholds = DEFAULT_THRESHOLDS) -> str:
    """A categoria que a regra dá para estas características, pelo nome."""
    return curate(f.boxes, f.clip, f.sharpness, f.width, f.height, thresholds).category.name


def cache_key() -> str:
    """Identifica "o que os modelos são": pesos, frases e confiança do detector."""
    payload = {
        "weights": [w.sha256 for w in weights.FILES],
        "prompts": flat_prompts(),
        "detector_min_score": config.DETECTOR_MIN_SCORE,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _to_json(f: Features) -> dict:
    return {
        "boxes": [[b.x0, b.y0, b.x1, b.y1, b.score] for b in f.boxes],
        "clip": f.clip,
        "sharpness": f.sharpness,
        "width": f.width,
        "height": f.height,
    }


def _from_json(d: dict) -> Features:
    return Features(
        boxes=tuple(Box(*b) for b in d["boxes"]),
        clip={k: float(v) for k, v in d["clip"].items()},
        sharpness=float(d["sharpness"]),
        width=int(d["width"]),
        height=int(d["height"]),
    )


def _load_cache(path: Path) -> dict[str, Features]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("key") != cache_key():
        return {}
    return {photo_id: _from_json(d) for photo_id, d in data["photos"].items()}


def _save_cache(path: Path, cached: dict[str, Features]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"key": cache_key(), "photos": {k: _to_json(v) for k, v in sorted(cached.items())}}
    path.write_text(json.dumps(data), encoding="utf-8")


def compute_all(
    rows: Sequence[Row], folder: Path, detector: Detector, zeroshot: ZeroShot, cache_path: Path
) -> dict[str, Features]:
    """As características de cada linha, do cache ou rodando os modelos.

    Uma foto sem arquivo é ERRO (FileNotFoundError com o id), nunca é pulada:
    pular mudaria a métrica sem ninguém perceber.
    """
    cached = _load_cache(cache_path)
    out: dict[str, Features] = {}
    for r in rows:
        if r.photo_id not in cached:
            path = image_path(folder, r)
            if not path.is_file():
                raise FileNotFoundError(f"a foto {r.photo_id} do manifesto não tem arquivo em {path}")
            with Image.open(path) as original:
                image = original.convert("RGB")
            boxes = tuple(detector.detect(image))
            cached[r.photo_id] = Features(
                boxes=boxes,
                clip=dict(zeroshot.classify(image)),
                sharpness=measure(image, largest_box(boxes)),
                width=image.width,
                height=image.height,
            )
        out[r.photo_id] = cached[r.photo_id]
    _save_cache(cache_path, cached)
    return out


def load_models(models_dir: Path) -> tuple[Detector, ZeroShot]:
    """Confere os pesos e carrega os dois modelos (alguns segundos)."""
    # Import aqui dentro: torch e transformers só carregam quando há modelo de
    # verdade (os testes deste módulo usam modelos falsos).
    from photo_curator.detector import RtDetrDetector
    from photo_curator.zeroshot import ClipZeroShot

    weights.verify(models_dir)
    return RtDetrDetector(models_dir / weights.RTDETR_SUBDIR), ClipZeroShot(models_dir / weights.CLIP_FILE)
