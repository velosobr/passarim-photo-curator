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


def half_of(halves: dict[str, list[Row]]) -> dict[str, str]:
    return {r.photo_id: name for name, members in halves.items() for r in members}


def test_frozen_observations_keep_their_half_when_set_grows():
    # Crescer o conjunto não pode mudar a metade das fotos que já foram sorteadas
    # (senão uma foto vista na calibração passaria para a medição).
    before = sp.split(dataset())
    frozen = sp.freeze(before)
    grown = dataset() + [row(900 + i, 5000 + i, "SINGLE_BIRD") for i in range(7)]
    after = half_of(sp.split(grown, frozen=frozen))
    for photo_id, name in half_of(before).items():
        assert after[photo_id] == name
    new = [after[str(900 + i)] for i in range(7)]
    assert sp.MEASUREMENT in new and sp.CALIBRATION in new


def test_frozen_observation_keeps_half_after_relabel():
    before = sp.split(dataset())
    frozen = sp.freeze(before)
    relabeled = [
        row(int(r.photo_id), int(r.observation_id), "FLOCK") if r.observation_id == "1000" else r
        for r in dataset()
    ]
    assert half_of(sp.split(relabeled, frozen=frozen)) == half_of(before)


def test_frozen_assignment_round_trips_through_file(tmp_path):
    frozen = sp.freeze(sp.split(dataset()))
    sp.write_frozen(tmp_path / "split.json", frozen)
    assert sp.read_frozen(tmp_path / "split.json") == frozen
    assert sp.read_frozen(tmp_path / "missing.json") == {}
