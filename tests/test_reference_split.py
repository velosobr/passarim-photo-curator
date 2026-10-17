"""Testes da divisão do conjunto de referência em calibração e medição."""

from photo_curator.reference import split as sp
from photo_curator.reference.manifest import EXCLUDED, Row


def row(photo_id: int, observation_id: int, label: str) -> Row:
    return Row(str(photo_id), str(observation_id), "sp", label, label, "a", "cc-by", "u", "species")


def dataset() -> list[Row]:
    # 30 observações de ave sozinha (2 fotos cada), 20 de ninho (1 foto) e 5 de bando.
    rows, pid = [], 0
    for obs in range(30):
        for _ in range(2):
            pid += 1
            rows.append(row(pid, 1000 + obs, "SINGLE_BIRD"))
    for obs in range(20):
        pid += 1
        rows.append(row(pid, 2000 + obs, "NEST"))
    for obs in range(5):
        pid += 1
        rows.append(row(pid, 3000 + obs, "FLOCK"))
    return rows


def test_split_is_deterministic():
    assert sp.split(dataset()) == sp.split(dataset())


def test_same_observation_never_in_both_halves():
    halves = sp.split(dataset())
    cal = {r.observation_id for r in halves[sp.CALIBRATION]}
    mea = {r.observation_id for r in halves[sp.MEASUREMENT]}
    assert not cal & mea
    assert len(halves[sp.CALIBRATION]) + len(halves[sp.MEASUREMENT]) == len(dataset())


def test_split_is_stratified_by_label():
    halves = sp.split(dataset())
    for label in ("SINGLE_BIRD", "NEST", "FLOCK"):
        groups = {half: {r.observation_id for r in rows if r.label == label} for half, rows in halves.items()}
        # A metade de medição fica com metade dos grupos (ou um a mais, se ímpar).
        assert len(groups[sp.MEASUREMENT]) - len(groups[sp.CALIBRATION]) in (0, 1)


def test_other_seed_gives_other_split():
    a = sp.split(dataset(), seed=1)[sp.MEASUREMENT]
    b = sp.split(dataset(), seed=2)[sp.MEASUREMENT]
    assert a != b


def test_excluded_and_unlabeled_rows_are_left_out():
    rows = [row(1, 1, "SINGLE_BIRD"), row(2, 2, EXCLUDED), row(3, 3, "")]
    halves = sp.split(rows)
    assert [r.photo_id for h in halves.values() for r in h] == ["1"]


def test_check_minimum_accepts_enough_photos():
    rows = [row(i, i, "SINGLE_BIRD") for i in range(50)] + [row(100 + i, 100 + i, "EGGS") for i in range(40)]
    assert sp.check_minimum(rows) == []


def test_check_minimum_reports_shortfall():
    rows = [row(i, i, "SINGLE_BIRD") for i in range(49)] + [row(100 + i, 100 + i, "FLOCK") for i in range(40)]
    problems = sp.check_minimum(rows)
    assert len(problems) == 2  # 49 aves sozinhas e 0 descartes (bando não é descarte)
    assert "49" in problems[0]
