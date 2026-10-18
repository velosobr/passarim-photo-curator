"""Calibra os limites do config.py na metade de CALIBRAÇÃO.

Uso: CURATOR_MODELS_DIR=models uv run python scripts/calibrate.py

Imprime o relatório antes e depois e o trecho para colar no config.py. Não
olha a metade de medição (ver reference/split.py). As saídas dos modelos ficam
em .cache/reference-features.json: a segunda execução é instantânea.
"""

import os
import sys
from pathlib import Path

from photo_curator.config import DEFAULT_THRESHOLDS
from photo_curator.reference import calibrate, features, manifest, split

REFERENCE = Path("tests/reference")
CACHE = Path(".cache/reference-features.json")


def main() -> int:
    models_dir = os.environ.get("CURATOR_MODELS_DIR")
    if not models_dir:
        sys.exit("defina CURATOR_MODELS_DIR (pasta dos pesos)")
    rows = manifest.read(REFERENCE / "manifest.csv")
    # As observações já sorteadas ficam na metade gravada em split.json.
    frozen = split.read_frozen(REFERENCE / "split.json")
    half = split.split(rows, frozen=frozen)[split.CALIBRATION]
    detector, zeroshot = features.load_models(Path(models_dir))
    feats = features.compute_all(half, REFERENCE, detector, zeroshot, CACHE)
    samples = [(r.label, feats[r.photo_id]) for r in half]

    print(f"calibração: {len(samples)} fotos\n\n## Limites atuais\n")
    print(calibrate.report_for(samples, DEFAULT_THRESHOLDS).markdown())
    best, report = calibrate.search(samples)
    print("## Limites calibrados\n")
    print(report.markdown())
    print("## Para o config.py\n")
    print(calibrate.config_snippet(best))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
