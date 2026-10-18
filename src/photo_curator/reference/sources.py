"""De onde vêm as fotos do conjunto de referência.

Duas fontes:
- as espécies do Passarim (scripts/species.txt): as fotos mais votadas, que são
  justamente as que o worker vai ver. A sugestão é SINGLE_BIRD;
- os CASOS DIFÍCEIS, buscados de propósito pelas anotações (ave morta, ovo,
  pena, ninho, osso) e pela descrição (anilhamento = ave na mão; bando).

A sugestão (proposed_label) vem da busca, NUNCA do curador: se viesse do
modelo, o dono tenderia a confirmar o que o modelo disse, e a métrica ficaria
otimista. Muita foto de "ninho" mostra o adulto fora dele: o dono corrige.

Para crescer o conjunto (se a medição não tiver o tamanho mínimo), aumente os
números de observações daqui e rode o build_reference.py de novo: as fotos já
baixadas e os rótulos já revisados ficam.

Quem usa: scripts/build_reference.py.
"""

from collections.abc import Mapping
from dataclasses import dataclass

AVES_TAXON_ID = "3"  # "Aves" no iNaturalist: a classe inteira


@dataclass(frozen=True)
class Source:
    name: str  # vai para a coluna "source" do manifesto
    params: Mapping[str, str]  # filtro acrescentado à busca de observações
    proposed_label: str
    observations: int  # quantas observações pedir
    photos_per_observation: int


# Por espécie: 4 observações, 1 foto de cada (a primeira, que é a que o worker
# vê primeiro). 41 espécies dão até 164 fotos. Começou com 2; subiu para 4 na
# montagem (2026-10-17) para a medição ter o mínimo de 50 aves sozinhas.
SPECIES_OBSERVATIONS = 4
SPECIES_PHOTOS = 1


def species_source(taxon_id: str, observations: int = SPECIES_OBSERVATIONS) -> Source:
    # "research": identificação confirmada pela comunidade (como no catalog).
    return Source(
        "species",
        {"taxon_id": taxon_id, "quality_grade": "research"},
        "SINGLE_BIRD",
        observations,
        SPECIES_PHOTOS,
    )


def _hard(name: str, label: str, observations: int, **params: str) -> Source:
    # Casos difíceis: qualquer ave, até 2 fotos por observação (a 2ª foto de uma
    # observação de ninho costuma ser o ninho, e a 1ª, o adulto).
    return Source(name, {"taxon_id": AVES_TAXON_ID, **params}, label, observations, 2)


# Números aumentados na montagem (2026-10-17) para a medição ter o mínimo de
# 40 descartes: muita foto de "ninho" ou "morto" mostra uma ave sozinha.
HARD_SOURCES: tuple[Source, ...] = (
    _hard("dead", "DEAD_BIRD", 18, term_id="17", term_value_id="19"),
    _hard("egg", "EGGS", 9, term_id="1", term_value_id="7"),
    _hard("egg", "EGGS", 9, term_id="22", term_value_id="30"),
    _hard("feather", "FEATHER", 18, term_id="22", term_value_id="23"),
    _hard("nest", "NEST", 20, term_id="22", term_value_id="35"),
    _hard("bone", "DEAD_BIRD", 6, term_id="22", term_value_id="27"),
    _hard("in_hand", "OTHER", 4, q="anilhamento", search_on="description"),
    _hard("in_hand", "OTHER", 12, q="banding", search_on="description"),
    _hard("flock", "FLOCK", 12, q="bando", search_on="description"),
)
