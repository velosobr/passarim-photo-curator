"""Busca de fotos no iNaturalist para montar o conjunto de referência.

O iNaturalist é uma rede de observações de natureza: cada OBSERVAÇÃO é um
encontro com um ser vivo (data, lugar, espécie) e tem uma ou mais FOTOS. A API
é aberta, mas os termos pedem um User-Agent que identifique o projeto e no
máximo ~1 requisição por segundo: o Client cuida das duas coisas.

As ANOTAÇÕES são marcações padronizadas que a comunidade põe nas observações
(ex.: "vivo ou morto: morto", "evidência: ninho"). Usamos elas para achar de
propósito os casos difíceis (ver sources.py).

Usamos só a biblioteca padrão (urllib) para não acrescentar dependência: são
poucas centenas de requisições, uma vez.

Quem usa: scripts/build_reference.py.
"""

import json
import math
import time
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from urllib.parse import urlencode

from photo_curator.reference.manifest import Row

API = "https://api.inaturalist.org/v1"
USER_AGENT = "passarim-photo-curator (+https://github.com/velosobr/passarim-photo-curator)"
BRAZIL_PLACE_ID = "6878"  # id do Brasil no iNaturalist (o mesmo do catalog)
# Licenças da política do Passarim (as mesmas do catalog). ND nunca.
LICENSES: tuple[str, ...] = ("cc-by", "cc-by-nc", "cc-by-sa", "cc-by-nc-sa", "cc0")
# Intervalo mínimo entre requisições, em segundos (um pouco acima de 1 s).
MIN_INTERVAL = 1.1
# Teto do tamanho de uma resposta: resposta externa nunca é confiável.
_MAX_BYTES = 20 << 20


@dataclass(frozen=True)
class RemotePhoto:
    """Uma foto candidata, como veio da API (antes de baixar)."""

    photo_id: str
    observation_id: str
    species: str
    author: str
    license: str
    url: str  # página da observação (para a atribuição)
    image_url: str  # arquivo da foto, na versão "large" (1024 px)


def author_of(attribution: str) -> str:
    """Extrai o nome do autor do texto de atribuição da API.

    '(c) desuni, some rights reserved (CC BY-NC)' -> 'desuni'
    """
    text = attribution.strip()
    if text.lower().startswith("(c)"):
        text = text[3:].strip()
    for tail in (", some rights reserved", ", no rights reserved", ", all rights reserved"):
        i = text.lower().find(tail)
        if i >= 0:
            text = text[:i]
    return text.strip() or "desconhecido"


def observation_params(extra: Mapping[str, str], per_page: int) -> dict[str, str]:
    """Parâmetros da busca de observações: sempre no Brasil, com foto, com
    licença aceita e ordenadas pelos votos (as fotos mais "favoritadas" vêm
    primeiro, como no catalog). `extra` acrescenta o filtro de cada busca."""
    params = {
        "place_id": BRAZIL_PLACE_ID,
        "photos": "true",
        "photo_license": ",".join(LICENSES),
        "order_by": "votes",
        "per_page": str(per_page),
    }
    params.update(extra)
    return params


def photos_of(observation: Mapping, max_photos: int) -> list[RemotePhoto]:
    """As fotos aceitas de uma observação, até max_photos. A licença é conferida
    foto a foto: o filtro da API vale para a observação, e uma observação aceita
    pode ter outra foto com licença ND."""
    species = (observation.get("taxon") or {}).get("name", "")
    page = observation.get("uri") or f"https://www.inaturalist.org/observations/{observation['id']}"
    out: list[RemotePhoto] = []
    for p in observation.get("photos", []):
        if len(out) == max_photos:
            break
        if p.get("license_code") not in LICENSES:
            continue
        out.append(
            RemotePhoto(
                photo_id=str(p["id"]),
                observation_id=str(observation["id"]),
                species=species,
                author=author_of(p.get("attribution", "")),
                license=p["license_code"],
                url=page,
                # A API devolve a miniatura quadrada; a "large" tem o mesmo caminho.
                image_url=p["url"].replace("square.", "large.", 1),
            )
        )
    return out


def add_photo(rows: dict[str, Row], photo: RemotePhoto, proposed_label: str, source: str) -> bool:
    """Acrescenta a foto ao manifesto (em memória), se ela ainda não estiver lá.

    Devolve False se já estava: uma foto pode vir de duas buscas, e rodar o
    script de novo não pode apagar um rótulo que o dono já revisou.
    """
    if photo.photo_id in rows:
        return False
    rows[photo.photo_id] = Row(
        photo_id=photo.photo_id,
        observation_id=photo.observation_id,
        species=photo.species,
        proposed_label=proposed_label,
        label="",
        author=photo.author,
        license=photo.license,
        url=photo.url,
        source=source,
    )
    return True


class Client:
    """Cliente mínimo da API, com User-Agent e ritmo de ~1 requisição por segundo.

    sleep e clock podem ser trocados nos testes (para não esperar de verdade).
    time.monotonic é um relógio que só anda para a frente, bom para medir
    intervalos (o relógio do sistema pode ser ajustado e "voltar").
    """

    def __init__(
        self,
        user_agent: str = USER_AGENT,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._user_agent = user_agent
        self._sleep = sleep
        self._clock = clock
        self._last = -math.inf

    def _wait(self) -> None:
        """Espera o que falta para completar MIN_INTERVAL desde a última requisição."""
        remaining = MIN_INTERVAL - (self._clock() - self._last)
        if remaining > 0:
            self._sleep(remaining)
        self._last = self._clock()

    def _open(self, url: str) -> bytes:
        self._wait()
        request = urllib.request.Request(url, headers={"User-Agent": self._user_agent})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read(_MAX_BYTES + 1)
        if len(data) > _MAX_BYTES:
            raise OSError(f"resposta maior que {_MAX_BYTES} bytes: {url}")
        return data

    def get_json(self, path: str, params: Mapping[str, str]) -> dict:
        return json.loads(self._open(f"{API}{path}?{urlencode(params)}"))

    def get_bytes(self, url: str) -> bytes:
        return self._open(url)

    def taxon_id(self, scientific_name: str) -> str | None:
        """Id do táxon pelo nome científico. A busca é "autocomplete", então só
        aceitamos o primeiro resultado se o nome bater exatamente (como o catalog)."""
        data = self.get_json("/taxa", {"q": scientific_name, "rank": "species", "per_page": "1"})
        results = data.get("results", [])
        if results and results[0].get("name", "").lower() == scientific_name.lower():
            return str(results[0]["id"])
        return None
