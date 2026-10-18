"""Divide o conjunto de referência em duas metades: calibração e medição.

Por que duas metades: se escolhêssemos os limites do config.py olhando as
mesmas fotos em que medimos o resultado, a medição ficaria otimista (os limites
"decorariam" aquelas fotos). Então os limites saem da metade de calibração, e a
CI mede na outra, que a calibração nunca vê.

Como dividimos:
- AGRUPADA por observação: as fotos de uma mesma observação (a mesma ave, no
  mesmo dia) ficam sempre na mesma metade. Senão a calibração veria uma foto
  quase igual a outra da medição, e a medição vazaria.
- ESTRATIFICADA pelo rótulo: dentro de cada rótulo, metade dos grupos vai para
  cada lado. Assim as duas metades têm proporções parecidas de ninho, ovo,
  bando etc., em vez de o sorteio jogar todos os ovos de um lado só.
- SEMENTE fixa: o embaralhamento usa um gerador de números "aleatórios" que,
  com a mesma semente, produz sempre a mesma sequência. A divisão sai igual em
  qualquer máquina e em qualquer dia.

- CONGELADA depois do primeiro sorteio: tests/reference/split.json guarda a
  metade de cada observação já sorteada. Quando o conjunto cresce, ou o dono
  troca o rótulo de uma foto, o embaralhamento de um rótulo mudaria inteiro (a
  lista ficou maior, ou a observação mudou de rótulo), e fotos já vistas na
  calibração cairiam na medição. Com o arquivo, as antigas ficam onde estavam
  e só as observações novas são sorteadas, entre elas.

A MEDIÇÃO NÃO AJUSTA LIMITES. Se alguém mudar um limite depois de olhar o
resultado da medição, precisa trocar SEED (e registrar o motivo no commit): o
sorteio novo cria uma medição que a pessoa ainda não viu.

Quem usa: scripts/calibrate.py e scripts/measure.py (leem split.json) e
scripts/apply_review.py (grava split.json depois de aplicar a revisão).
"""

import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from photo_curator.reference.manifest import DISCARD_LABELS, EXCLUDED, Row

SEED = 20261017
CALIBRATION = "calibration"
MEASUREMENT = "measurement"

# Tamanho mínimo da metade de medição (spec 4.5): com menos fotos, uma foto a
# mais ou a menos mexe demais na porcentagem.
MIN_SINGLE = 50
MIN_DISCARD = 40


def _group_label(members: Sequence[Row]) -> str:
    """O rótulo de um grupo (observação) é o mais comum entre as fotos dele; no
    empate, o primeiro em ordem alfabética, para o resultado ser sempre o mesmo."""
    counts = Counter(r.label for r in members)
    return min(counts, key=lambda label: (-counts[label], label))


def split(
    rows: Sequence[Row], seed: int = SEED, frozen: Mapping[str, str] | None = None
) -> dict[str, list[Row]]:
    """Divide as fotos rotuladas (sem as EXCLUDED e as ainda sem rótulo).

    `frozen` (observation_id -> metade) é o sorteio já feito, lido de
    split.json: essas observações ficam na metade gravada, e só as que não
    estão nele passam pelo sorteio abaixo.
    """
    frozen = frozen or {}
    usable = [r for r in rows if r.label and r.label != EXCLUDED]
    groups: dict[str, list[Row]] = defaultdict(list)
    for r in usable:
        groups[r.observation_id].append(r)

    halves: dict[str, list[Row]] = {CALIBRATION: [], MEASUREMENT: []}
    strata: dict[str, list[str]] = defaultdict(list)
    for observation_id, members in groups.items():
        if frozen.get(observation_id) in halves:
            halves[frozen[observation_id]].extend(members)
        else:
            strata[_group_label(members)].append(observation_id)

    for label in sorted(strata):
        # Ordenar antes de embaralhar: o resultado não pode depender da ordem
        # das linhas no CSV. A semente leva o rótulo junto para cada estrato ter
        # o seu próprio sorteio.
        observation_ids = sorted(strata[label], key=int)
        random.Random(f"{seed}:{label}").shuffle(observation_ids)
        for i, observation_id in enumerate(observation_ids):
            # Alterna: 1º grupo para a medição, 2º para a calibração... Com um
            # número ímpar de grupos, a medição fica com um a mais.
            half = MEASUREMENT if i % 2 == 0 else CALIBRATION
            halves[half].extend(groups[observation_id])
    for members in halves.values():
        members.sort(key=lambda r: int(r.photo_id))
    return halves


def freeze(halves: Mapping[str, Sequence[Row]]) -> dict[str, str]:
    """O sorteio feito, no formato de split.json: observation_id -> metade."""
    return {r.observation_id: name for name, members in halves.items() for r in members}


def read_frozen(path: Path) -> dict[str, str]:
    """Lê split.json; sem o arquivo (primeiro sorteio), nada está congelado."""
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def write_frozen(path: Path, frozen: Mapping[str, str]) -> None:
    """Grava split.json em ordem numérica, para o diff do git ficar legível."""
    ordered = {k: frozen[k] for k in sorted(frozen, key=int)}
    path.write_text(json.dumps(ordered, indent=1) + "\n", encoding="utf-8")


def check_minimum(measurement: Sequence[Row]) -> list[str]:
    """Os problemas de tamanho da metade de medição (lista vazia = tudo certo)."""
    single = sum(r.label == "SINGLE_BIRD" for r in measurement)
    discard = sum(r.label in DISCARD_LABELS for r in measurement)
    problems = []
    if single < MIN_SINGLE:
        problems.append(f"a medição tem {single} fotos SINGLE_BIRD; o mínimo é {MIN_SINGLE}")
    if discard < MIN_DISCARD:
        problems.append(f"a medição tem {discard} fotos de descarte; o mínimo é {MIN_DISCARD}")
    return problems
