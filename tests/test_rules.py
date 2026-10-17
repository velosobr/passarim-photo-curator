"""Testes da regra de decisão (photo_curator.rules.curate).

A regra é uma função pura: recebe o que os modelos viram (caixas do detector,
probabilidades do CLIP, nitidez) e devolve o veredito. Por isso estes testes não
carregam modelo nenhum e rodam em milissegundos. Cada caso é uma foto
"imaginada" com números escolhidos para cair num ramo da regra.
"""

import pytest

from photo_curator import config
from photo_curator.rules import clamp_box, curate, largest_box
from photo_curator.types import CLIP_CLASSES, Box, Category

W, H = 800, 600  # tamanho de foto usado na maioria dos casos


def clip(**probs: float) -> dict[str, float]:
    """Monta as probabilidades do CLIP: as classes não citadas ficam em 0."""
    scores = dict.fromkeys(CLIP_CLASSES, 0.0)
    scores.update(probs)
    return scores


def bird(x0: float, y0: float, x1: float, y1: float, score: float = 0.9) -> Box:
    return Box(x0, y0, x1, y1, score)


def test_single_bird_in_the_middle():
    v = curate([bird(200, 150, 600, 450)], clip(single=0.9), 0.7, W, H)
    assert v.category is Category.SINGLE_BIRD
    assert v.bird_count == 1
    assert v.largest_bird_area == pytest.approx(400 * 300 / (W * H))
    assert v.largest_bird_cut is False
    assert v.sharpness == 0.7
    assert v.confidence == pytest.approx(0.9)  # min(CLIP single 0,9; detector 0,9)


def test_nest_with_chick_is_nest_even_with_one_bird():
    # O filhote é "uma ave" para o detector, mas a foto é de ninho: o descarte vem antes.
    v = curate([bird(300, 200, 500, 400)], clip(nest=0.8, single=0.2), 0.5, W, H)
    assert v.category is Category.NEST
    assert v.confidence == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("clip_class", "expected"),
    [
        ("eggs", Category.EGGS),
        ("feather", Category.FEATHER),
        ("dead", Category.DEAD_BIRD),
        ("in_hand", Category.OTHER),
        ("drawing", Category.OTHER),
    ],
)
def test_content_discards(clip_class, expected):
    v = curate([bird(300, 200, 500, 400)], clip(**{clip_class: 0.9}), 0.5, W, H)
    assert v.category is expected


def test_flock_without_detections_is_flock():
    # Caso real da andorinha: o detector não acha nenhuma ave (pequenas demais),
    # mas o CLIP tem certeza de que é um bando. Tem que ser FLOCK, não OTHER.
    v = curate([], clip(flock=1.0), 0.4, W, H)
    assert v.category is Category.FLOCK
    assert v.bird_count == 0
    assert v.confidence == pytest.approx(1.0)


def test_flock_by_clip_with_one_detected_bird():
    v = curate([bird(10, 10, 60, 60)], clip(flock=0.9, single=0.1), 0.4, W, H)
    assert v.category is Category.FLOCK


def test_no_bird_when_clip_agrees():
    v = curate([], clip(no_bird=0.7), 0.3, W, H)
    assert v.category is Category.NO_BIRD
    assert v.confidence == pytest.approx(0.7)


def test_no_detection_without_clip_agreement_is_other():
    v = curate([], clip(single=0.6, no_bird=0.2), 0.3, W, H)
    assert v.category is Category.OTHER
    assert v.confidence == pytest.approx(0.8)  # 1 - no_bird


def test_strong_no_bird_with_weak_detection_is_no_bird():
    v = curate([bird(0, 0, 50, 50, score=0.55)], clip(no_bird=0.9), 0.3, W, H)
    assert v.category is Category.NO_BIRD


def test_strong_no_bird_with_confident_detection_keeps_bird():
    v = curate([bird(200, 150, 600, 450, score=0.95)], clip(no_bird=0.9, single=0.1), 0.3, W, H)
    assert v.category is Category.SINGLE_BIRD


def test_couple_side_by_side_is_flock():
    v = curate([bird(100, 100, 300, 400), bird(400, 100, 620, 400)], clip(single=0.5, flock=0.3), 0.5, W, H)
    assert v.category is Category.FLOCK
    assert v.bird_count == 2


def test_dominant_bird_with_small_one_behind_is_single():
    # A maior ocupa 300x300 = 90 000 px²; a do fundo, 100x100 = 10 000 px² (9x menor).
    v = curate([bird(100, 100, 400, 400), bird(600, 50, 700, 150, score=0.6)], clip(single=0.8), 0.6, W, H)
    assert v.category is Category.SINGLE_BIRD
    assert v.bird_count == 2


def test_dominance_threshold_is_respected():
    # Segunda ave com exatamente 1/DOMINANCE da área: ainda conta como dominante.
    big = bird(0, 0, 400, 400)  # 160 000
    side = (160_000 / config.DOMINANCE) ** 0.5
    small = bird(500, 0, 500 + side, side)
    v = curate([big, small], clip(single=0.8), 0.6, W, H)
    assert v.category is Category.SINGLE_BIRD


def test_close_up_touching_edge_is_single_and_cut():
    # Caso real da arara-azul: a caixa vai de x=0 a 713 numa foto de 706 px de largura.
    v = curate([bird(0, 53, 713, 703, score=0.97)], clip(single=1.0), 0.8, 706, 800)
    assert v.category is Category.SINGLE_BIRD
    assert v.largest_bird_cut is True
    assert v.largest_bird_area <= 1.0


def test_bird_inside_edge_tolerance_is_cut():
    margin = config.EDGE_TOLERANCE * W / 2  # metade da tolerância: encosta
    v = curate([bird(margin, 200, 400, 400)], clip(single=0.9), 0.5, W, H)
    assert v.largest_bird_cut is True


def test_bird_outside_edge_tolerance_is_not_cut():
    margin = config.EDGE_TOLERANCE * W * 2  # o dobro da tolerância: não encosta
    v = curate([bird(margin, 200, 400, 400)], clip(single=0.9), 0.5, W, H)
    assert v.largest_bird_cut is False


def test_box_outside_image_is_clamped():
    box = clamp_box(Box(-20, -10, 900, 700, 0.9), W, H)
    assert (box.x0, box.y0, box.x1, box.y1) == (0, 0, W, H)
    v = curate([Box(-20, -10, 900, 700, 0.9)], clip(single=0.9), 0.5, W, H)
    assert v.largest_bird_area == pytest.approx(1.0)


def test_largest_box_picks_biggest_area():
    small, big = bird(0, 0, 10, 10), bird(0, 0, 100, 100)
    assert largest_box([small, big]) is big
    assert largest_box([]) is None
