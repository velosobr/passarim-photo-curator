"""A regra de decisão: transforma o que os modelos viram num veredito.

É o coração do curador, e de propósito é uma função pura: não carrega modelo,
não lê arquivo, não acessa rede. Recebe números e devolve um Verdict. Isso
permite testar cada ramo com casos inventados (tests/test_rules.py).

A ordem dos passos importa e está explicada em curate().
Quem usa: server.py. Depende de: config.py e types.py.
"""

from collections.abc import Mapping, Sequence

from photo_curator import config
from photo_curator.types import Box, Category, Verdict

# Classes do CLIP que descartam a foto, e a categoria que cada uma vira.
_DISCARD_CATEGORY: dict[str, Category] = {
    "nest": Category.NEST,
    "eggs": Category.EGGS,
    "feather": Category.FEATHER,
    "dead": Category.DEAD_BIRD,
    "in_hand": Category.OTHER,
    "drawing": Category.OTHER,
}


def clamp_box(box: Box, width: int, height: int) -> Box:
    """Prende a caixa dentro da foto. O detector às vezes devolve coordenadas um
    pouco fora (negativas ou maiores que a foto); sem isso a área passaria de 1."""
    return Box(
        x0=min(max(box.x0, 0.0), width),
        y0=min(max(box.y0, 0.0), height),
        x1=min(max(box.x1, 0.0), width),
        y1=min(max(box.y1, 0.0), height),
        score=box.score,
    )


def largest_box(boxes: Sequence[Box]) -> Box | None:
    """A caixa de maior área, ou None se não houver nenhuma."""
    return max(boxes, key=lambda b: b.area, default=None)


def _touches_edge(box: Box, width: int, height: int) -> bool:
    """A caixa encosta na borda? "Encostar" é ficar a menos de EDGE_TOLERANCE
    (2%) do lado da foto, porque o detector raramente acerta o pixel exato."""
    mx, my = config.EDGE_TOLERANCE * width, config.EDGE_TOLERANCE * height
    return box.x0 <= mx or box.y0 <= my or box.x1 >= width - mx or box.y1 >= height - my


def _has_dominant(boxes: Sequence[Box]) -> bool:
    """Com duas aves ou mais, a maior é "dominante" se tem pelo menos DOMINANCE
    vezes a área da segunda maior."""
    areas = sorted((b.area for b in boxes), reverse=True)
    return areas[0] >= config.DOMINANCE * areas[1]


def curate(
    boxes: Sequence[Box],
    clip: Mapping[str, float],
    sharpness: float,
    width: int,
    height: int,
) -> Verdict:
    """Decide a categoria de uma foto.

    boxes: as aves achadas pelo detector (já filtradas por DETECTOR_MIN_SCORE).
    clip: probabilidade de cada classe do CLIP (somam 1).
    sharpness: nitidez da maior ave (ou da foto), de 0 a 1.
    width, height: tamanho da foto em pixels.

    A ordem dos passos:
    1. Descarte por conteúdo (ninho, ovos, pena, morta, na mão, ilustração). Vem
       primeiro porque um ninho com filhote também tem "uma ave" para o detector.
    2. Bando pelo CLIP. Vem antes do "sem ave" porque, num bando no fio, as aves
       são pequenas demais e o detector não acha nenhuma (caso real da andorinha).
    3. Nenhuma ave detectada: NO_BIRD se o CLIP concorda, senão OTHER.
    4. Detecção falsa: CLIP com "sem ave" muito forte e detecção fraca -> NO_BIRD.
    5. Duas aves ou mais sem uma dominante: FLOCK (um casal também).
    6. O que sobra: SINGLE_BIRD.
    """
    clamped = [clamp_box(b, width, height) for b in boxes]
    biggest = largest_box(clamped)
    photo_area = float(width * height)

    def verdict(category: Category, confidence: float) -> Verdict:
        # Os campos de tamanho e corte vêm sempre da maior ave, seja qual for a
        # categoria: o worker usa isso na pontuação e no relatório.
        return Verdict(
            category=category,
            bird_count=len(clamped),
            largest_bird_area=biggest.area / photo_area if biggest else 0.0,
            largest_bird_cut=_touches_edge(biggest, width, height) if biggest else False,
            sharpness=sharpness,
            confidence=float(min(max(confidence, 0.0), 1.0)),
        )

    # 1. Descarte por conteúdo: vence a classe de maior probabilidade entre as
    # que passaram do limite.
    passed = [
        (clip.get(name, 0.0), name)
        for name in _DISCARD_CATEGORY
        if clip.get(name, 0.0) >= config.DISCARD_MIN[name]
    ]
    if passed:
        prob, name = max(passed)
        return verdict(_DISCARD_CATEGORY[name], prob)

    # 2. Bando pelo CLIP.
    if clip.get("flock", 0.0) >= config.FLOCK_MIN:
        return verdict(Category.FLOCK, clip["flock"])

    # 3. Nenhuma ave detectada.
    if biggest is None:
        no_bird = clip.get("no_bird", 0.0)
        if no_bird >= config.NO_BIRD_MIN:
            return verdict(Category.NO_BIRD, no_bird)
        # OTHER: o CLIP acha que há algo além de paisagem; a confiança é o quanto
        # ele NÃO acha que é paisagem.
        return verdict(Category.OTHER, 1.0 - no_bird)

    # 4. Detecção falsa.
    best_score = max(b.score for b in clamped)
    if clip.get("no_bird", 0.0) >= config.NO_BIRD_STRONG and best_score < config.WEAK_DETECTION:
        return verdict(Category.NO_BIRD, clip["no_bird"])

    # 5. Bando pelo detector.
    if len(clamped) >= 2 and not _has_dominant(clamped):
        return verdict(Category.FLOCK, min(clip.get("flock", 0.0) + clip.get("single", 0.0), biggest.score))

    # 6. Ave sozinha: a confiança é a menor entre o CLIP ("single") e o detector.
    return verdict(Category.SINGLE_BIRD, min(clip.get("single", 0.0), biggest.score))
