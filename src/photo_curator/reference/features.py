"""As "características" de cada foto: o que os modelos viram, guardado em cache.

Rodar o detector e o CLIP leva ~0,5 s por foto. A regra (rules.curate), ao
contrário, é instantânea. Então rodamos os modelos UMA vez por foto, guardamos
as saídas (caixas, probabilidades do CLIP, nitidez e tamanho) num arquivo
JSON, e a calibração experimenta milhares de combinações de limites só com a
regra.

O cache tem uma CHAVE: o hash dos pesos, das frases do CLIP e da confiança
mínima do detector. Se qualquer um mudar, as saídas mudam, e o cache antigo é
ignorado (senão a calibração usaria saídas de outro modelo). Cada foto no
cache também guarda o hash do ARQUIVO dela: se a foto for trocada no disco, as
saídas antigas não valem mais e os modelos rodam de novo só para ela.

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
    """Tudo o que a regra precisa saber de uma foto, já calculado pelos modelos.

    - boxes: as caixas de ave que o detector achou (com a confiança de cada uma);
    - clip: a probabilidade que o CLIP deu a cada classe de frase (ninho, ovo,
      bando...), somando 1 entre todas;
    - sharpness: a nitidez (0 a 1) medida na maior caixa, ou na foto inteira;
    - width e height: o tamanho da foto em pixels.
    """

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


def _file_hash(path: Path) -> str:
    """O sha256 do arquivo da foto: muda se qualquer byte da foto mudar."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_cache(path: Path) -> dict[str, tuple[str, Features]]:
    """O cache em disco: photo_id -> (hash do arquivo, características).

    Se a chave gravada não bate com a de agora (outros pesos, outras frases), o
    cache inteiro é de outro modelo: devolvemos vazio, e tudo é recalculado. Não
    é erro, é só o caminho mais lento.
    """
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("key") != cache_key():
        return {}
    return {photo_id: (d.get("file_sha256", ""), _from_json(d)) for photo_id, d in data["photos"].items()}


def _save_cache(path: Path, cached: dict[str, tuple[str, Features]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    photos = {k: {**_to_json(f), "file_sha256": h} for k, (h, f) in sorted(cached.items())}
    path.write_text(json.dumps({"key": cache_key(), "photos": photos}), encoding="utf-8")


def _run_models(photo_id: str, path: Path, detector: Detector, zeroshot: ZeroShot) -> Features:
    """Abre a foto e roda os dois modelos nela.

    Uma foto que não abre (arquivo cortado por um download interrompido, ou que
    não é imagem) vira ValueError com o id: a mensagem do PIL sozinha ("could
    not create decoder object") não diz qual das centenas de fotos é.
    """
    try:
        with Image.open(path) as original:
            image = original.convert("RGB")
    except (OSError, Image.DecompressionBombError) as err:
        raise ValueError(f"a foto {photo_id} ({path}) não abre: {err}") from err
    boxes = tuple(detector.detect(image))
    return Features(
        boxes=boxes,
        clip=dict(zeroshot.classify(image)),
        sharpness=measure(image, largest_box(boxes)),
        width=image.width,
        height=image.height,
    )


def compute_all(
    rows: Sequence[Row], folder: Path, detector: Detector, zeroshot: ZeroShot, cache_path: Path
) -> dict[str, Features]:
    """As características de cada linha, do cache ou rodando os modelos.

    Uma foto sem arquivo é ERRO (FileNotFoundError com o id), nunca é pulada,
    mesmo que ela esteja no cache: pular mudaria a métrica sem ninguém perceber.
    Uma foto que não abre também é erro (ValueError com o id).
    """
    cached = _load_cache(cache_path)
    out: dict[str, Features] = {}
    for r in rows:
        path = image_path(folder, r)
        if not path.is_file():
            raise FileNotFoundError(f"a foto {r.photo_id} do manifesto não tem arquivo em {path}")
        file_hash = _file_hash(path)
        entry = cached.get(r.photo_id)
        # Só aproveita o cache se o arquivo for o mesmo que gerou aquelas saídas.
        if entry is None or entry[0] != file_hash:
            cached[r.photo_id] = (file_hash, _run_models(r.photo_id, path, detector, zeroshot))
        out[r.photo_id] = cached[r.photo_id][1]
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
