"""Self-check da imagem Docker: `python -m photo_curator.selfcheck <pasta>`.

Roda DENTRO do container, de preferência com `--network none --read-only`.
Prova duas coisas de uma vez:
- a imagem tem tudo para classificar sem internet (pesos, bibliotecas);
- as três fotos reais saem com as categorias certas, pelo gRPC de verdade.

A pasta tem andorinha-bando.webp, anu-ovos.webp e arara-sozinha.webp (as
fixtures dos testes, montadas no container como volume só de leitura).
"""

import os
import sys
import time
from pathlib import Path

import grpc
from passarim.curator.v1 import curator_pb2, curator_pb2_grpc

from photo_curator import weights
from photo_curator.__main__ import Settings
from photo_curator.detector import RtDetrDetector
from photo_curator.server import CuratorServicer, build_server
from photo_curator.zeroshot import ClipZeroShot

EXPECTED = {
    "andorinha-bando.webp": curator_pb2.PHOTO_CATEGORY_FLOCK,
    "anu-ovos.webp": curator_pb2.PHOTO_CATEGORY_EGGS,
    "arara-sozinha.webp": curator_pb2.PHOTO_CATEGORY_SINGLE_BIRD,
}


def main(folder: Path) -> int:
    # MODELS_DIR vale aqui também: na imagem é /opt/models; fora dela, a pasta local.
    models = Settings.from_env(os.environ).models_dir
    weights.verify(models)
    servicer = CuratorServicer(
        RtDetrDetector(models / weights.RTDETR_SUBDIR), ClipZeroShot(models / weights.CLIP_FILE)
    )
    server, port = build_server(servicer, "127.0.0.1:0")
    server.start()
    try:
        photos = [
            curator_pb2.PhotoInput(photo_id=name, image=(folder / name).read_bytes()) for name in EXPECTED
        ]
        # Repete até 8 fotos: o mesmo tamanho de lote que o worker manda.
        photos = (photos * 3)[:8]
        start = time.perf_counter()
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            resp = curator_pb2_grpc.PhotoCuratorServiceStub(channel).ClassifyPhotos(
                curator_pb2.ClassifyPhotosRequest(photos=photos), timeout=60
            )
        elapsed = time.perf_counter() - start
    finally:
        server.stop(grace=None)
    failures = [
        f"{v.photo_id}: {curator_pb2.PhotoCategory.Name(v.category)}"
        for v in resp.verdicts
        if v.category != EXPECTED[v.photo_id]
    ]
    print(f"lote de {len(photos)} fotos em {elapsed:.1f} s; modelo {resp.model_version}")
    if failures:
        print("categorias erradas:", *failures, sep="\n  ")
        return 1
    if elapsed > 15:
        print("lento demais: o limite é 15 s por lote de 8")
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("uso: python -m photo_curator.selfcheck <pasta com as fotos de teste>")
    sys.exit(main(Path(sys.argv[1])))
