"""Mede o curador na metade de MEDIÇÃO e reprova abaixo das metas.

Uso: CURATOR_MODELS_DIR=models uv run python scripts/measure.py

Usa os limites do config.py (os do serviço). Escreve o resumo na saída e, na
CI, também em $GITHUB_STEP_SUMMARY (o resumo que aparece na página do job do
GitHub Actions). Sai com 1 se a medição for pequena demais ou se uma meta não
for atingida.
"""

import os
import sys
from pathlib import Path

from photo_curator import config
from photo_curator.reference import features, manifest, metrics, split

REFERENCE = Path("tests/reference")
CACHE = Path(".cache/reference-features.json")


def main() -> int:
    models_dir = os.environ.get("CURATOR_MODELS_DIR")
    if not models_dir:
        sys.exit("defina CURATOR_MODELS_DIR (pasta dos pesos)")
    rows = manifest.read(REFERENCE / "manifest.csv")
    half = split.split(rows)[split.MEASUREMENT]
    problems = split.check_minimum(half)
    if problems:
        print("medição pequena demais:", *problems, sep="\n  ")
        return 1

    detector, zeroshot = features.load_models(Path(models_dir))
    feats = features.compute_all(half, REFERENCE, detector, zeroshot, CACHE)
    report = metrics.evaluate((r.label, features.predict(feats[r.photo_id])) for r in half)

    text = (
        f"## Curador na metade de medição ({len(half)} fotos, {config.MODEL_VERSION})\n\n{report.markdown()}"
    )
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(text)
    failures = report.failures()
    if failures:
        print("abaixo das metas:", *failures, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
