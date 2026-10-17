"""As frases do CLIP para cada classe e a soma das probabilidades por classe.

Como o CLIP classifica "sem treino" (zero-shot): ele transforma a foto e cada
frase em vetores, num mesmo "espaço", e mede o quanto a foto se parece com cada
frase. Aplicando softmax nessas semelhanças (uma conta que transforma números
quaisquer em probabilidades que somam 1), temos a probabilidade de cada frase.

Cada classe tem algumas frases, para não depender de uma palavra só. A
probabilidade da classe é a SOMA das probabilidades das suas frases, num
softmax único sobre todas as frases de todas as classes.

As frases são em inglês porque o CLIP foi treinado com legendas em inglês.
O plano A2 pode ajustá-las na calibração; mudar uma frase muda MODEL_VERSION.

Quem usa: zeroshot.py (o modelo de verdade) e os testes.
"""

from collections.abc import Sequence

from photo_curator.types import CLIP_CLASSES

PROMPTS: dict[str, tuple[str, ...]] = {
    "single": ("a photo of a single bird", "a close-up photo of a bird perched"),
    "flock": ("a photo of a flock of birds", "many birds on a wire"),
    "nest": ("a bird nest", "a nest with chicks"),
    "eggs": ("bird eggs in a nest", "a clutch of eggs"),
    "feather": ("a single feather on the ground", "a bird feather"),
    "dead": ("a dead bird", "a bird carcass"),
    "in_hand": ("a bird held in a human hand", "a bird being banded"),
    "drawing": ("a drawing or illustration of a bird", "a photo of a camera screen"),
    "no_bird": ("a photo of a landscape with no animals", "a photo of trees"),
}


def flat_prompts() -> list[str]:
    """Todas as frases numa lista só, na ordem de CLIP_CLASSES (é a ordem em
    que o CLIP as recebe e devolve as probabilidades)."""
    return [text for name in CLIP_CLASSES for text in PROMPTS[name]]


def aggregate(prompt_probs: Sequence[float]) -> dict[str, float]:
    """Soma as probabilidades das frases de cada classe.

    prompt_probs vem do softmax sobre flat_prompts(), na mesma ordem.
    """
    expected = len(flat_prompts())
    if len(prompt_probs) != expected:
        raise ValueError(f"esperava {expected} probabilidades, recebi {len(prompt_probs)}")
    scores: dict[str, float] = {}
    i = 0
    for name in CLIP_CLASSES:
        n = len(PROMPTS[name])
        scores[name] = float(sum(prompt_probs[i : i + n]))
        i += n
    return scores
