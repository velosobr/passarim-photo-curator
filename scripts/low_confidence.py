"""Imprime a sugestão de LOW_CONFIDENCE para o worker do catalog.

Uso: CURATOR_MODELS_DIR=models uv run python scripts/low_confidence.py

Roda a regra (com os limites do config.py) nas fotos da metade de CALIBRAÇÃO
do conjunto de referência, pega a confiança das que saíram SINGLE_BIRD e
imprime o percentil 20 (photo_curator.reference.confidence). Usa a metade de
calibração, e não a de medição, pela mesma razão da calibração dos limites:
a medição só mede, não ajusta nada.

Também conta quantos falsos positivos (fotos que o curador chamou de ave
sozinha, mas que não são) ficariam marcados: é o que a marca quer pegar.
"""

import os
import sys
from pathlib import Path

from photo_curator.reference import features, manifest, split
from photo_curator.reference.confidence import suggest_low_confidence
from photo_curator.rules import curate

REFERENCE = Path("tests/reference")
CACHE = Path(".cache/reference-features.json")


def main() -> int:
    models_dir = os.environ.get("CURATOR_MODELS_DIR")
    if not models_dir:
        sys.exit("defina CURATOR_MODELS_DIR (pasta dos pesos)")
    rows = manifest.read(REFERENCE / "manifest.csv")
    frozen = split.read_frozen(REFERENCE / "split.json")
    half = split.split(rows, frozen=frozen)[split.CALIBRATION]

    detector, zeroshot = features.load_models(Path(models_dir))
    feats = features.compute_all(half, REFERENCE, detector, zeroshot, CACHE)

    # (rótulo revisado, confiança) de cada foto que a regra chamou de SINGLE_BIRD.
    singles = []
    for r in half:
        f = feats[r.photo_id]
        v = curate(f.boxes, f.clip, f.sharpness, f.width, f.height)
        if v.category.name == "SINGLE_BIRD":
            singles.append((r.label, v.confidence))

    limit = suggest_low_confidence([c for _, c in singles])
    false_pos = [c for label, c in singles if label != "SINGLE_BIRD"]
    caught = sum(1 for c in false_pos if c <= limit)
    print(f"fotos SINGLE_BIRD na calibração: {len(singles)}")
    print(f"falsos positivos: {len(false_pos)}, {caught} deles marcados com o limite")
    print(f"LOW_CONFIDENCE sugerido: {limit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
