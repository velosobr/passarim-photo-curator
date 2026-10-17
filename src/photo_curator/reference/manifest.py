"""O manifesto do conjunto de referência: tests/reference/manifest.csv.

CSV ("valores separados por vírgula") é uma tabela em texto: a primeira linha
tem os nomes das colunas e cada linha seguinte é uma foto. Escolhemos CSV porque
dá para ler e revisar no GitHub, e o diff de um PR mostra exatamente quais
rótulos mudaram.

Colunas:
- photo_id, observation_id: ids no iNaturalist (a observação agrupa as fotos
  de um mesmo encontro com a ave);
- species: nome científico que a observação traz;
- proposed_label: a sugestão de rótulo que veio da busca (ver sources.py);
- label: o rótulo revisado pelo dono, que vale para as métricas (vazio até a revisão);
- author, license, url: atribuição exigida pela licença CC;
- source: de qual busca a foto veio (species, dead, nest...).

Quem usa: os scripts de scripts/ e os outros módulos de photo_curator.reference.
"""

import csv
from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path

# Os rótulos possíveis são as categorias do contrato, menos UNSPECIFIED (que é
# "a imagem não abriu", nunca um rótulo de verdade).
LABELS: tuple[str, ...] = (
    "SINGLE_BIRD",
    "FLOCK",
    "NEST",
    "EGGS",
    "FEATHER",
    "DEAD_BIRD",
    "NO_BIRD",
    "OTHER",
)
# EXCLUDED: o dono marcou "ignorar" (foto ambígua). Fica no manifesto, mas sai
# da divisão e das métricas.
EXCLUDED = "EXCLUDED"
# Os rótulos de descarte, contados juntos na meta de "revocação dos descartes".
DISCARD_LABELS = frozenset({"NEST", "EGGS", "FEATHER", "DEAD_BIRD", "NO_BIRD", "OTHER"})


@dataclass(frozen=True)
class Row:
    """Uma foto do conjunto de referência (uma linha do CSV)."""

    photo_id: str
    observation_id: str
    species: str
    proposed_label: str
    label: str
    author: str
    license: str
    url: str
    source: str


# A ordem das colunas no arquivo é a ordem dos campos da Row.
COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(Row))


def read(path: Path) -> list[Row]:
    """Lê o manifesto. Recusa um arquivo com colunas diferentes das esperadas,
    em vez de ler colunas trocadas sem perceber."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"colunas de {path}: {reader.fieldnames}; esperado {list(COLUMNS)}")
        return [Row(**line) for line in reader]


def write(path: Path, rows: Iterable[Row]) -> None:
    """Grava o manifesto em ordem numérica de photo_id: a mesma entrada dá
    sempre o mesmo arquivo, e o diff do PR mostra só o que mudou."""
    ordered = sorted(rows, key=lambda r: int(r.photo_id))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(COLUMNS)
        for r in ordered:
            writer.writerow([getattr(r, c) for c in COLUMNS])


def image_path(folder: Path, row: Row) -> Path:
    """Onde fica a foto de uma linha: <pasta>/photos/<photo_id>.webp."""
    return folder / "photos" / f"{row.photo_id}.webp"


def attribution_markdown(rows: Iterable[Row]) -> str:
    """O texto do ATTRIBUTION.md: a licença CC exige citar autor, licença e
    origem de cada foto."""
    lines = [
        "# Fotos do conjunto de referência",
        "",
        "As fotos de `photos/` vêm do iNaturalist e **mantêm a licença de cada uma** (fora da licença do",
        "repositório). Foram reduzidas a 512 px no lado maior e convertidas para WebP.",
        "",
        "| Arquivo | Autor | Licença | Observação |",
        "|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: int(r.photo_id)):
        lines.append(f"| `photos/{r.photo_id}.webp` | {r.author} | {r.license} | {r.url} |")
    return "\n".join(lines) + "\n"
