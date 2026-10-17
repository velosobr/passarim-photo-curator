"""Monta a página de revisão em build/review/, pronta para publicar como Artifact.

Uso: uv run python scripts/make_review_page.py [--reference tests/reference]

Gera:
- build/review/index.html: cópia de review/index.html (a página);
- build/review/reference.json: as fotos do manifesto, com a sugestão e o
  rótulo atual (a página lê este arquivo);
- build/review/photos/<photo_id>.webp: cópia das fotos.

A publicação é feita pelo executor com a ferramenta Artifact, declarando a
capacidade "db" (banco compartilhado, onde a página grava os rótulos).
"""

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from photo_curator.reference import manifest

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, default=ROOT / "tests" / "reference")
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "review")
    args = parser.parse_args()

    rows = manifest.read(args.reference / "manifest.csv")
    if args.out.exists():
        shutil.rmtree(args.out)  # sempre do zero: nada de foto velha sobrando
    (args.out / "photos").mkdir(parents=True)
    shutil.copyfile(ROOT / "review" / "index.html", args.out / "index.html")
    for r in rows:
        shutil.copyfile(manifest.image_path(args.reference, r), args.out / "photos" / f"{r.photo_id}.webp")
    (args.out / "reference.json").write_text(
        json.dumps([asdict(r) for r in rows], ensure_ascii=False), encoding="utf-8"
    )
    print(f"{len(rows)} fotos em {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
