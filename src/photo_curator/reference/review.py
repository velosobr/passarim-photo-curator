"""Aplica a revisão do dono ao manifesto.

O dono rotula as fotos na página de revisão (review/index.html). O executor
exporta o banco da página para tests/reference/review.json, no formato
{"<photo_id>": "<RÓTULO>"}, e este módulo copia cada rótulo para a coluna
"label" do manifesto.

Recusa id ou rótulo desconhecido em vez de gravar: um erro na exportação não
pode virar silenciosamente um rótulo errado na métrica.

Quem usa: scripts/apply_review.py.
"""

from collections.abc import Mapping, Sequence
from dataclasses import replace

from photo_curator.reference.manifest import EXCLUDED, LABELS, Row

_VALID = frozenset({*LABELS, EXCLUDED})


def apply(rows: Sequence[Row], decisions: Mapping[str, str]) -> list[Row]:
    """Devolve as linhas com o rótulo da revisão; as que a revisão não cita
    ficam com o rótulo que já tinham."""
    unknown = sorted(set(decisions) - {r.photo_id for r in rows})
    if unknown:
        raise ValueError(f"fotos que não estão no manifesto: {', '.join(unknown)}")
    bad = sorted({label for label in decisions.values() if label not in _VALID})
    if bad:
        raise ValueError(f"rótulos desconhecidos: {', '.join(bad)}")
    return [replace(r, label=decisions.get(r.photo_id, r.label)) for r in rows]


def missing(rows: Sequence[Row]) -> list[str]:
    """Os ids das fotos ainda sem rótulo."""
    return [r.photo_id for r in rows if not r.label]
