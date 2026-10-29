"""Sugere o limite de "baixa confiança" que o worker do catalog usa no relatório.

Quem usa: scripts/low_confidence.py. O valor impresso vira o padrão de
LOW_CONFIDENCE no passarim-catalog (internal/config/worker.go).

Para que serve: quando a foto principal escolhida pelo modelo tem confiança
abaixo desse limite, o worker marca a espécie com "low_confidence" no
relatório de curadoria, e o dono confere aquela foto primeiro. Não há um
número "certo" de antemão: a confiança depende de como a regra junta o
detector e o CLIP, e a maioria das aves sozinhas sai com confiança alta. Por
isso o limite sai dos dados, da metade de calibração do conjunto de referência.
"""

import math
from collections.abc import Sequence


def suggest_low_confidence(confidences: Sequence[float], share: float = 0.2) -> float:
    """Devolve o "percentil" `share` das confianças: o valor que deixa essa fração
    das fotos abaixo dele ou nele.

    confidences: a confiança das fotos que o curador chamou de SINGLE_BIRD.
    Com share = 0,2, as 20% menos confiantes ficam marcadas para conferência.
    """
    if not confidences:
        raise ValueError("nenhuma foto SINGLE_BIRD na metade de calibração")
    ordered = sorted(confidences)
    # ceil(0,2 × 10) = 2: a 2ª menor (índice 1). max(0, ...) cobre a lista curta.
    index = max(0, math.ceil(share * len(ordered)) - 1)
    return round(ordered[index], 3)
