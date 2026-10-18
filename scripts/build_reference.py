"""Monta (ou cresce) o conjunto de referência: baixa fotos do iNaturalist.

Uso: uv run python scripts/build_reference.py [--out tests/reference]

Para cada busca de reference/sources.py, pede as observações, pega as fotos com
licença aceita, reduz a 512 px (WebP) e acrescenta ao manifesto com a sugestão
de rótulo e o rótulo vazio. Rodar de novo só acrescenta fotos novas: as que já
estão no manifesto (e os rótulos já revisados) ficam como estão.

Leva alguns minutos: a API pede ~1 requisição por segundo, e cada foto é uma
requisição.
"""

import argparse
from pathlib import Path

from photo_curator.reference import images, inat, manifest, sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("tests/reference"))
    parser.add_argument("--species-observations", type=int, default=sources.SPECIES_OBSERVATIONS)
    args = parser.parse_args()

    (args.out / "photos").mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.csv"
    rows = {r.photo_id: r for r in manifest.read(manifest_path)} if manifest_path.exists() else {}
    client = inat.Client()

    names = [
        line.strip()
        for line in (Path(__file__).parent / "species.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    all_sources: list[sources.Source] = []
    for name in names:
        taxon_id = client.taxon_id(name)
        if taxon_id is None:
            print(f"aviso: {name} não encontrada no iNaturalist; pulando")
            continue
        all_sources.append(sources.species_source(taxon_id, args.species_observations))
    all_sources.extend(sources.HARD_SOURCES)

    for source in all_sources:
        data = client.get_json("/observations", inat.observation_params(source.params, source.observations))
        added = 0
        for observation in data.get("results", []):
            for photo in inat.photos_of(observation, source.photos_per_observation):
                if photo.photo_id in rows:
                    continue
                target = args.out / "photos" / f"{photo.photo_id}.webp"
                try:
                    if not target.exists():
                        # Grava num .tmp e só depois renomeia: se o download for
                        # interrompido no meio, não sobra um .webp cortado que a
                        # próxima rodada tomaria por foto pronta.
                        partial = target.with_suffix(".tmp")
                        partial.write_bytes(images.shrink(client.get_bytes(photo.image_url)))
                        partial.replace(target)
                except OSError as err:
                    # Foto que não baixa ou não abre fica de fora (e não entra no
                    # manifesto, para nunca haver linha sem arquivo).
                    print(f"aviso: foto {photo.photo_id} falhou ({err}); pulando")
                    continue
                added += inat.add_photo(rows, photo, source.proposed_label, source.name)
        print(f"{source.name} {dict(source.params)}: +{added} (total {len(rows)})")

    manifest.write(manifest_path, rows.values())
    (args.out / "ATTRIBUTION.md").write_text(manifest.attribution_markdown(rows.values()), encoding="utf-8")
    print(f"{len(rows)} fotos em {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
