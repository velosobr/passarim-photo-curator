"""As interfaces dos modelos, separadas da implementação.

Protocol (do módulo typing) descreve "o que um objeto precisa saber fazer",
sem herança: qualquer classe com um método detect(image) -> list[Box] serve
como Detector. O servidor depende só dessas interfaces. Em produção, quem as
cumpre são detector.py e zeroshot.py; nos testes, classes falsas e rápidas.
"""

from typing import Protocol

from PIL import Image

from photo_curator.types import Box


class Detector(Protocol):
    def detect(self, image: Image.Image) -> list[Box]:
        """Devolve as caixas das aves (só classe "bird", já filtradas pela
        confiança mínima), em pixels da imagem recebida."""
        ...


class ZeroShot(Protocol):
    def classify(self, image: Image.Image) -> dict[str, float]:
        """Devolve a probabilidade de cada classe de CLIP_CLASSES (somam 1)."""
        ...
