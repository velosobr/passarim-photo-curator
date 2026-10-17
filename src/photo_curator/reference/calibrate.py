"""A busca dos limites da regra na metade de calibração.

BUSCA COORDENADA: em vez de testar todas as combinações de 11 limites (seriam
trilhões), mexemos em UM limite por vez. Para cada limite, testamos os valores
da grade com os outros parados, ficamos com o melhor e passamos ao próximo.
Damos duas voltas completas, porque mudar um limite pode mudar qual é o melhor
valor de outro.

"Melhor", nesta ordem:
1. o menor déficit das metas (0 = todas atingidas, ver Report.deficit);
2. a maior revocação de SINGLE_BIRD (mais fotos boas aproveitadas);
3. a maior acurácia geral;
4. a menor distância dos limites padrão, para não mexer sem motivo.

Isto roda SÓ na metade de calibração. Rodar na medição e copiar os limites
seria "decorar" a prova (ver reference/split.py).

Quem usa: scripts/calibrate.py.
"""

from collections.abc import Sequence
from dataclasses import replace

from photo_curator.config import DEFAULT_THRESHOLDS, Thresholds
from photo_curator.reference.features import Features, predict
from photo_curator.reference.metrics import Report, evaluate


def _steps(start: float, stop: float, step: float) -> tuple[float, ...]:
    """Valores de start a stop (inclusive), arredondados para não acumular erro
    de ponto flutuante (0,1 + 0,2 não é exatamente 0,3 no computador)."""
    n = round((stop - start) / step)
    return tuple(round(start + i * step, 4) for i in range(n + 1))


_PROB = _steps(0.2, 0.9, 0.05)
_DISCARD_CLASSES = tuple(DEFAULT_THRESHOLDS.discard_min)

# Os limites calibrados. "discard_min.<classe>" é o limite de uma classe de descarte.
PARAMS: tuple[str, ...] = (
    *(f"discard_min.{c}" for c in _DISCARD_CLASSES),
    "flock_min",
    "no_bird_min",
    "no_bird_strong",
    "weak_detection",
    "dominance",
)
GRID: dict[str, tuple[float, ...]] = {
    **{f"discard_min.{c}": _PROB for c in _DISCARD_CLASSES},
    "flock_min": _PROB,
    "no_bird_min": _PROB,
    "no_bird_strong": _steps(0.6, 0.95, 0.05),
    "weak_detection": _steps(0.5, 0.9, 0.05),
    "dominance": (2.0, 3.0, 4.0, 5.0, 6.0, 8.0),
}


def _get(t: Thresholds, param: str) -> float:
    if param.startswith("discard_min."):
        return t.discard_min[param.split(".", 1)[1]]
    return getattr(t, param)


def with_param(t: Thresholds, param: str, value: float) -> Thresholds:
    """Uma cópia de t com um limite trocado (t não muda: Thresholds é imutável)."""
    if param.startswith("discard_min."):
        discard = dict(t.discard_min)
        discard[param.split(".", 1)[1]] = value
        return replace(t, discard_min=discard)
    return replace(t, **{param: value})


def report_for(samples: Sequence[tuple[str, Features]], t: Thresholds) -> Report:
    """As métricas dos limites t nas amostras (rótulo, características)."""
    return evaluate((label, predict(f, t)) for label, f in samples)


def _distance(t: Thresholds) -> float:
    return sum(abs(_get(t, p) - _get(DEFAULT_THRESHOLDS, p)) for p in PARAMS)


def _rank(t: Thresholds, report: Report) -> tuple[float, float, float, float]:
    # max() escolhe a maior tupla; por isso os itens "quanto menor, melhor" vão negativos.
    return (-report.deficit(), report.single_recall.value, report.accuracy.value, -_distance(t))


def search(
    samples: Sequence[tuple[str, Features]], start: Thresholds = DEFAULT_THRESHOLDS, passes: int = 2
) -> tuple[Thresholds, Report]:
    """Os melhores limites (e o relatório deles) pela busca coordenada."""
    best = start
    best_report = report_for(samples, best)
    for _ in range(passes):
        for param in PARAMS:
            for value in GRID[param]:
                candidate = with_param(best, param, value)
                report = report_for(samples, candidate)
                if _rank(candidate, report) > _rank(best, best_report):
                    best, best_report = candidate, report
    return best, best_report


def config_snippet(t: Thresholds) -> str:
    """O trecho para colar no config.py, no lugar das constantes atuais."""
    discard = ",\n".join(f'    "{c}": {t.discard_min[c]}' for c in _DISCARD_CLASSES)
    return (
        f"DISCARD_MIN: dict[str, float] = {{\n{discard},\n}}\n"
        f"FLOCK_MIN = {t.flock_min}\n"
        f"NO_BIRD_MIN = {t.no_bird_min}\n"
        f"NO_BIRD_STRONG = {t.no_bird_strong}\n"
        f"WEAK_DETECTION = {t.weak_detection}\n"
        f"DOMINANCE = {t.dominance}\n"
    )
