"""As métricas do conjunto de referência (spec, seção 1).

- Precisão de SINGLE_BIRD: das fotos que o curador chamou de ave sozinha,
  quantas são mesmo. É a meta mais importante: erro aqui põe ninho ou bando
  como foto principal.
- Revocação de SINGLE_BIRD: das fotos que são ave sozinha, quantas o curador
  reconheceu. Sem ela, um curador que descarta tudo passaria nas outras metas.
- Revocação dos descartes: das fotos de ninho, ovos, pena, ave morta, sem ave
  ou "outro", quantas o curador tirou da escolha. Conta como pega qualquer
  predição que não seja SINGLE_BIRD nem FLOCK (UNSPECIFIED também tira a foto).

INTERVALO DE WILSON: com ~90 fotos, a porcentagem medida é uma estimativa. O
intervalo de 95% diz a faixa onde o valor "de verdade" provavelmente está. A
fórmula de Wilson funciona bem com poucas amostras e perto de 0% ou 100%, onde
a fórmula simples (p ± 1,96·√(p(1-p)/n)) passaria de 100%.

MATRIZ DE CONFUSÃO: uma tabela com o rótulo nas linhas e a predição nas
colunas. A diagonal são os acertos; o resto mostra QUE erro o curador comete
(ex.: ovos vistos como ave sozinha).

Quem usa: reference/calibrate.py e scripts/measure.py.
"""

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from photo_curator.reference.manifest import DISCARD_LABELS, LABELS

# Metas da spec. Os nomes são os campos do Report.
TARGETS: dict[str, float] = {
    "single_precision": 0.95,
    "single_recall": 0.70,
    "discard_recall": 0.90,
}

_TITLES = {
    "single_precision": "Precisão de SINGLE_BIRD",
    "single_recall": "Revocação de SINGLE_BIRD",
    "discard_recall": "Revocação dos descartes",
}

# Predições que tiram a foto da escolha da foto principal e da galeria.
_KEPT = frozenset({"SINGLE_BIRD", "FLOCK"})


def wilson(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de Wilson (z = 1,96 dá 95%). Sem amostras: (0, 1)."""
    if total == 0:
        return (0.0, 1.0)
    p = hits / total
    denominator = 1 + z * z / total
    center = p + z * z / (2 * total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (max(0.0, (center - margin) / denominator), min(1.0, (center + margin) / denominator))


@dataclass(frozen=True)
class Rate:
    """Uma proporção "acertos de total", com o valor e o intervalo."""

    hits: int
    total: int

    @property
    def value(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def interval(self) -> tuple[float, float]:
        return wilson(self.hits, self.total)


@dataclass(frozen=True)
class Report:
    single_precision: Rate
    single_recall: Rate
    discard_recall: Rate
    accuracy: Rate
    # (rótulo, predição) -> quantas fotos.
    confusion: dict[tuple[str, str], int]

    def failures(self) -> list[str]:
        """As metas não atingidas, em texto. Uma meta sem nenhuma foto conta como
        não atingida: passar "por vazio" esconderia um conjunto mal montado."""
        out = []
        for name, target in TARGETS.items():
            rate: Rate = getattr(self, name)
            if rate.total == 0 or rate.value < target:
                out.append(
                    f"{_TITLES[name].lower()}: {rate.value:.1%} ({rate.hits}/{rate.total}), meta {target:.0%}"
                )
        return out

    def deficit(self) -> float:
        """Quanto falta, somado, para atingir as metas (0 = todas atingidas). A
        calibração usa isso para preferir os limites que chegam mais perto."""
        return sum(max(0.0, target - getattr(self, name).value) for name, target in TARGETS.items())

    def markdown(self) -> str:
        """O resumo que a CI publica: metas com intervalo, acurácia e matriz."""
        lines = ["| Meta | Valor | IC 95% (Wilson) | Mínimo | |", "|---|---|---|---|---|"]
        for name, target in TARGETS.items():
            rate: Rate = getattr(self, name)
            lo, hi = rate.interval()
            ok = "ok" if rate.total and rate.value >= target else "abaixo"
            lines.append(
                f"| {_TITLES[name]} | {rate.value:.1%} ({rate.hits}/{rate.total}) | {lo:.1%} a {hi:.1%} "
                f"| {target:.0%} | {ok} |"
            )
        lines += [
            "",
            f"Acurácia geral: {self.accuracy.value:.1%} ({self.accuracy.hits}/{self.accuracy.total})",
            "",
        ]
        columns = [*LABELS, "UNSPECIFIED"]
        lines.append("| rótulo \\ predição | " + " | ".join(columns) + " |")
        lines.append("|---" * (len(columns) + 1) + "|")
        for label in LABELS:
            counts = [str(self.confusion.get((label, c), 0)) for c in columns]
            lines.append(f"| {label} | " + " | ".join(counts) + " |")
        return "\n".join(lines) + "\n"


def evaluate(pairs: Iterable[tuple[str, str]]) -> Report:
    """Calcula o relatório a partir de pares (rótulo, predição)."""
    pairs = list(pairs)
    predicted_single = [label for label, pred in pairs if pred == "SINGLE_BIRD"]
    labeled_single = [pred for label, pred in pairs if label == "SINGLE_BIRD"]
    discards = [pred for label, pred in pairs if label in DISCARD_LABELS]
    return Report(
        single_precision=Rate(
            sum(label == "SINGLE_BIRD" for label in predicted_single), len(predicted_single)
        ),
        single_recall=Rate(sum(pred == "SINGLE_BIRD" for pred in labeled_single), len(labeled_single)),
        discard_recall=Rate(sum(pred not in _KEPT for pred in discards), len(discards)),
        accuracy=Rate(sum(label == pred for label, pred in pairs), len(pairs)),
        confusion=dict(Counter(pairs)),
    )
