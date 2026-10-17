"""Ponto de entrada: `python -m photo_curator` sobe o servidor gRPC.

Ordem: lê a configuração das variáveis de ambiente, confere os pesos (hash),
carrega os modelos (alguns segundos) e só então abre a porta. Assim o health
check só responde depois que o serviço consegue de fato classificar.

Variáveis: PORT (padrão 50061) e MODELS_DIR (padrão /opt/models).
"""

import logging
import os
import signal
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from photo_curator import weights
from photo_curator.detector import RtDetrDetector
from photo_curator.server import CuratorServicer, build_server
from photo_curator.zeroshot import ClipZeroShot

log = logging.getLogger("photo_curator")


@dataclass(frozen=True)
class Settings:
    port: int
    models_dir: Path

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "Settings":
        return cls(port=int(env.get("PORT", "50061")), models_dir=Path(env.get("MODELS_DIR", "/opt/models")))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = Settings.from_env(os.environ)
    weights.verify(settings.models_dir)
    log.info("carregando modelos de %s", settings.models_dir)
    servicer = CuratorServicer(
        RtDetrDetector(settings.models_dir / weights.RTDETR_SUBDIR),
        ClipZeroShot(settings.models_dir / weights.CLIP_FILE),
    )
    server, port = build_server(servicer, f"0.0.0.0:{settings.port}")
    server.start()
    log.info("curador ouvindo na porta %d", port)
    # SIGTERM é o sinal que o Docker manda no "stop": paramos com 10 s de folga
    # para as chamadas em andamento terminarem.
    signal.signal(signal.SIGTERM, lambda *_: server.stop(grace=10))
    server.wait_for_termination()


if __name__ == "__main__":
    main()
