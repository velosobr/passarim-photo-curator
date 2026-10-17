"""Testes das partes puras da busca no iNaturalist (sem rede)."""

from photo_curator.reference import inat
from photo_curator.reference.manifest import Row

OBSERVATION = {
    "id": 408681516,
    "uri": "https://www.inaturalist.org/observations/408681516",
    "taxon": {"name": "Vanellus chilensis"},
    "photos": [
        {
            "id": 1,
            "license_code": "cc-by-nc",
            "attribution": "(c) desuni, some rights reserved (CC BY-NC)",
            "url": "https://static.inaturalist.org/photos/1/square.jpg",
        },
        {
            "id": 2,
            "license_code": "cc-by-nd",
            "attribution": "(c) x, some rights reserved (CC BY-ND)",
            "url": "https://static.inaturalist.org/photos/2/square.jpg",
        },
        {
            "id": 3,
            "license_code": None,
            "attribution": "(c) y, all rights reserved",
            "url": "https://static.inaturalist.org/photos/3/square.jpg",
        },
        {
            "id": 4,
            "license_code": "cc0",
            "attribution": "(c) Zé, no rights reserved (CC0)",
            "url": "https://static.inaturalist.org/photos/4/square.jpg",
        },
    ],
}


def test_author_of():
    assert inat.author_of("(c) desuni, some rights reserved (CC BY-NC)") == "desuni"
    assert inat.author_of("(c) Zé, no rights reserved (CC0)") == "Zé"
    assert inat.author_of("") == "desconhecido"


def test_photos_of_filters_licenses():
    photos = inat.photos_of(OBSERVATION, max_photos=5)
    assert [p.photo_id for p in photos] == ["1", "4"]  # ND e sem licença ficam de fora
    first = photos[0]
    assert first.observation_id == "408681516"
    assert first.species == "Vanellus chilensis"
    assert first.author == "desuni"
    assert first.image_url == "https://static.inaturalist.org/photos/1/large.jpg"
    assert first.url == OBSERVATION["uri"]


def test_photos_of_respects_max():
    assert len(inat.photos_of(OBSERVATION, max_photos=1)) == 1


def test_observation_params_fix_place_and_licenses():
    params = inat.observation_params({"term_id": "17"}, per_page=12)
    assert params["place_id"] == "6878"
    assert params["photo_license"] == "cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa,cc0"
    assert params["per_page"] == "12"
    assert params["term_id"] == "17"


def test_add_photo_keeps_existing_row():
    photo = inat.photos_of(OBSERVATION, max_photos=1)[0]
    reviewed = Row(
        "1", "408681516", "Vanellus chilensis", "NEST", "SINGLE_BIRD", "desuni", "cc-by-nc", "u", "nest"
    )
    rows = {"1": reviewed}
    # A mesma foto vinda de outra busca não sobrescreve o rótulo já revisado.
    assert inat.add_photo(rows, photo, "DEAD_BIRD", "dead") is False
    assert rows["1"] is reviewed
    rows.clear()
    assert inat.add_photo(rows, photo, "NEST", "nest") is True
    assert rows["1"].proposed_label == "NEST"
    assert rows["1"].label == ""
    assert rows["1"].source == "nest"


def test_client_waits_between_requests():
    # O relógio e o sleep são falsos: o teste não espera de verdade.
    now = [100.0]
    slept = []
    client = inat.Client(sleep=lambda s: slept.append(s), clock=lambda: now[0])
    client._wait()
    client._wait()  # chamada imediata: precisa esperar o intervalo inteiro
    assert slept == [inat.MIN_INTERVAL]
