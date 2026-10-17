"""O servidor gRPC do curador.

O que faz, para cada chamada ClassifyPhotos:
1. confere o tamanho do lote (1 a 8 fotos; fora disso, INVALID_ARGUMENT);
2. para cada foto: abre o WebP, roda o detector e o CLIP, mede a nitidez da
   maior ave e chama a regra (rules.curate);
3. devolve os vereditos na ordem do pedido.

Uma foto que não abre, que é grande demais (config.MAX_IMAGE_PIXELS) ou que faz
um modelo falhar vira PHOTO_CATEGORY_UNSPECIFIED e o lote segue: o erro é da
foto, não do lote, e derrubar tudo faria o worker tentar de novo para sempre.

gRPC em Python: o servidor tem um "pool" de threads; cada chamada roda numa
thread. Usamos 2 (o trabalho é de CPU e o worker manda um lote por vez); as
chamadas a mais esperam na fila do gRPC.

Também registra o health check padrão do gRPC (grpc.health.v1), que o compose
e o worker usam para saber se o curador está de pé.

Quem usa: __main__.py (produção) e tests/test_server.py.
Depende de: o contrato gerado (passarim.curator.v1), rules, sharpness, models.
"""

import io
import logging
from concurrent import futures

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from passarim.curator.v1 import curator_pb2, curator_pb2_grpc
from PIL import Image, UnidentifiedImageError

from photo_curator import config
from photo_curator.models import Detector, ZeroShot
from photo_curator.rules import curate, largest_box
from photo_curator.sharpness import measure
from photo_curator.types import Category, Verdict

log = logging.getLogger(__name__)

# Nome completo do serviço, como o health check o conhece.
SERVICE_NAME = curator_pb2.DESCRIPTOR.services_by_name["PhotoCuratorService"].full_name

_UNREADABLE = Verdict(Category.UNSPECIFIED, 0, 0.0, False, 0.0, 0.0)


class CuratorServicer(curator_pb2_grpc.PhotoCuratorServiceServicer):
    """Implementa o PhotoCuratorService. Herda da classe gerada, que já sabe
    receber e responder pelo gRPC; aqui só preenchemos o método."""

    def __init__(self, detector: Detector, zeroshot: ZeroShot, model_version: str = config.MODEL_VERSION):
        self._detector = detector
        self._zeroshot = zeroshot
        self._model_version = model_version

    def ClassifyPhotos(self, request, context):  # noqa: N802 (nome vem do contrato)
        n = len(request.photos)
        if not 1 <= n <= config.MAX_BATCH:
            # context.abort encerra a chamada com o código de erro do gRPC.
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, f"lote com {n} fotos; o limite é de 1 a {config.MAX_BATCH}"
            )
        verdicts = [self._classify_one(p.photo_id, p.image) for p in request.photos]
        return curator_pb2.ClassifyPhotosResponse(verdicts=verdicts, model_version=self._model_version)

    def _classify_one(self, photo_id: str, data: bytes) -> curator_pb2.PhotoVerdict:
        try:
            image = Image.open(io.BytesIO(data))
            # Image.open só lê o cabeçalho: o tamanho já é conhecido aqui, antes
            # de gastar memória decodificando os pixels.
            if image.width * image.height > config.MAX_IMAGE_PIXELS:
                log.warning(
                    "foto %s tem %dx%d px, acima do limite; veredito UNSPECIFIED", photo_id, *image.size
                )
                return _to_proto(photo_id, _UNREADABLE)
            image.load()  # força a leitura agora, para o erro aparecer aqui
            image = image.convert("RGB")
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
            log.warning("foto %s não abriu; veredito UNSPECIFIED", photo_id)
            return _to_proto(photo_id, _UNREADABLE)
        try:
            boxes = self._detector.detect(image)
            scores = self._zeroshot.classify(image)
            sharpness = measure(image, largest_box(boxes))
            return _to_proto(photo_id, curate(boxes, scores, sharpness, image.width, image.height))
        except Exception:
            # Qualquer erro nos modelos ou na regra também é "da foto": se ele
            # derrubasse o lote, o worker repetiria o mesmo lote e cairia de novo.
            # log.exception grava a pilha do erro para investigar depois.
            log.exception("erro ao classificar a foto %s; veredito UNSPECIFIED", photo_id)
            return _to_proto(photo_id, _UNREADABLE)


def _to_proto(photo_id: str, v: Verdict) -> curator_pb2.PhotoVerdict:
    """Converte o Verdict interno na mensagem do contrato. Category tem os mesmos
    números do enum do contrato, então int(v.category) basta."""
    return curator_pb2.PhotoVerdict(
        photo_id=photo_id,
        category=int(v.category),
        bird_count=v.bird_count,
        largest_bird_area=v.largest_bird_area,
        largest_bird_cut=v.largest_bird_cut,
        sharpness=v.sharpness,
        confidence=v.confidence,
    )


def build_server(servicer: CuratorServicer, address: str, max_workers: int = 2) -> tuple[grpc.Server, int]:
    """Monta o servidor (sem iniciar) e devolve (servidor, porta).

    address "0.0.0.0:50061" em produção; "127.0.0.1:0" nos testes (a porta 0
    pede ao sistema uma porta livre, devolvida por add_insecure_port).
    "insecure" = sem TLS: o curador só é acessível na rede interna do compose.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    curator_pb2_grpc.add_PhotoCuratorServiceServicer_to_server(servicer, server)
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    # "" é o status geral do servidor; SERVICE_NAME, o do nosso serviço.
    for name in ("", SERVICE_NAME):
        health_servicer.set(name, health_pb2.HealthCheckResponse.SERVING)
    port = server.add_insecure_port(address)
    return server, port
