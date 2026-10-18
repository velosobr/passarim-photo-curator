"""Copia a revisão do dono (tests/reference/review.json) para o manifesto.

Uso: uv run python scripts/apply_review.py [--reference tests/reference]

Sai com erro (1) se ainda houver foto sem rótulo, dizendo quantas: a divisão e
as métricas só valem com o conjunto inteiro revisado.

Com tudo revisado, congela a divisão em tests/reference/split.json: as
observações que já estavam lá ficam na mesma metade, e as novas são sorteadas
(ver reference/split.py).
"""

import argparse
import json
from pathlib import Path

from photo_curator.reference import manifest, review, split


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, default=Path("tests/reference"))
    args = parser.parse_args()

    path = args.reference / "manifest.csv"
    decisions = json.loads((args.reference / "review.json").read_text(encoding="utf-8"))
    rows = review.apply(manifest.read(path), decisions)
    manifest.write(path, rows)
    left = review.missing(rows)
    print(f"{len(rows) - len(left)} fotos rotuladas; {len(left)} sem rótulo")
    if left:
        print("sem rótulo:", ", ".join(left[:20]), "..." if len(left) > 20 else "")
        return 1
    split_path = args.reference / "split.json"
    halves = split.split(rows, frozen=split.read_frozen(split_path))
    split.write_frozen(split_path, split.freeze(halves))
    print(f"divisão congelada em {split_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
