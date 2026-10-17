"""Tipos que circulam dentro do curador.

Quem usa: rules.py (a decisão), server.py (converte de/para o contrato gRPC),
detector.py e zeroshot.py (produzem Box e as probabilidades do CLIP).

"dataclass(frozen=True)" cria uma classe só com dados, imutável: depois de
criada, ninguém muda os campos por engano no meio do caminho.
"""

import enum
from dataclasses import dataclass


class Category(enum.IntEnum):
    """O que a foto mostra. Os números são os mesmos do enum PhotoCategory do
    contrato (passarim.curator.v1), para a conversão ser direta: Category(3) é NEST.
    IntEnum é um enum que também se comporta como número inteiro."""

    UNSPECIFIED = 0
    SINGLE_BIRD = 1
    FLOCK = 2
    NEST = 3
    EGGS = 4
    FEATHER = 5
    DEAD_BIRD = 6
    NO_BIRD = 7
    OTHER = 8


# As "classes" do CLIP: cada uma tem algumas frases (ver prompts.py) e recebe
# uma probabilidade. Não são as mesmas categorias do contrato: "in_hand" e
# "drawing" viram OTHER, e "single" e "flock" ajudam a confirmar o detector.
CLIP_CLASSES: tuple[str, ...] = (
    "single",
    "flock",
    "nest",
    "eggs",
    "feather",
    "dead",
    "in_hand",
    "drawing",
    "no_bird",
)


@dataclass(frozen=True)
class Box:
    """A caixa de uma ave achada pelo detector, em pixels da foto: (x0, y0) é o
    canto de cima à esquerda e (x1, y1) o de baixo à direita. score é a confiança
    do detector, de 0 a 1."""

    x0: float
    y0: float
    x1: float
    y1: float
    score: float

    @property
    def area(self) -> float:
        """Área em pixels² (zero se a caixa estiver "virada")."""
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)


@dataclass(frozen=True)
class Verdict:
    """O veredito de uma foto; espelha o PhotoVerdict do contrato, sem o photo_id
    (quem conhece o id é o servidor)."""

    category: Category
    bird_count: int
    largest_bird_area: float
    largest_bird_cut: bool
    sharpness: float
    confidence: float
