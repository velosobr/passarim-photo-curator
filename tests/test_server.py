"""Testes do servidor gRPC com modelos FALSOS.

Por que falsos: os modelos de verdade levam ~1 s por foto e precisam de ~600 MB
de pesos. Aqui queremos testar o servidor (lote, ordem, erros, health check),
não os modelos. Os falsos devolvem respostas fixas escolhidas pelo tamanho da
imagem, que cada teste controla.

O servidor sobe de verdade numa porta livre do localhost, e o teste conversa
com ele pelo cliente gerado do contrato (passarim.curator.v1): isso também testa
o contrato de ponta a ponta.
"""

import io

import grpc
import pytest
from grpc_health.v1 import health_pb2, health_pb2_grpc
from passarim.curator.v1 import curator_pb2, curator_pb2_grpc
from PIL import Image

from photo_curator.server import CuratorServicer, build_server
from photo_curator.types import CLIP_CLASSES, Box


def webp(width: int, height: int) -> bytes:
    """Uma imagem WebP lisa do tamanho pedido (o tamanho é a "chave" dos falsos)."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (90, 120, 60)).save(buf, format="WEBP")
    return buf.getvalue()


def clip(**probs: float) -> dict[str, float]:
    scores = dict.fromkeys(CLIP_CLASSES, 0.0)
    scores.update(probs)
    return scores


class FakeDetector:
    """Detector falso: a resposta depende da largura da imagem."""

    def detect(self, image):
        return {
            400: [Box(100, 100, 300, 250, 0.95)],  # uma ave
            401: [],  # nada
        }.get(image.width, [])


class FakeZeroShot:
    def classify(self, image):
        return {
            400: clip(single=0.9),
            401: clip(eggs=0.95),
        }.get(image.width, clip(no_bird=0.9))


@pytest.fixture
def stub():
    server, port = build_server(CuratorServicer(FakeDetector(), FakeZeroShot(), "teste@1"), "127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    yield curator_pb2_grpc.PhotoCuratorServiceStub(channel), channel
    channel.close()
    server.stop(grace=None)


def request(*photos: tuple[str, bytes]) -> curator_pb2.ClassifyPhotosRequest:
    return curator_pb2.ClassifyPhotosRequest(
        photos=[curator_pb2.PhotoInput(photo_id=i, image=b) for i, b in photos]
    )


def test_classifies_in_request_order(stub):
    client, _ = stub
    resp = client.ClassifyPhotos(request(("a", webp(400, 300)), ("b", webp(401, 300))))
    assert resp.model_version == "teste@1"
    assert [v.photo_id for v in resp.verdicts] == ["a", "b"]
    assert resp.verdicts[0].category == curator_pb2.PHOTO_CATEGORY_SINGLE_BIRD
    assert resp.verdicts[0].bird_count == 1
    assert resp.verdicts[1].category == curator_pb2.PHOTO_CATEGORY_EGGS


def test_bad_image_does_not_break_batch(stub):
    client, _ = stub
    resp = client.ClassifyPhotos(
        request(("a", webp(400, 300)), ("ruim", b"isto nao e imagem"), ("c", webp(401, 300)))
    )
    assert [v.photo_id for v in resp.verdicts] == ["a", "ruim", "c"]
    assert resp.verdicts[1].category == curator_pb2.PHOTO_CATEGORY_UNSPECIFIED
    assert resp.verdicts[0].category == curator_pb2.PHOTO_CATEGORY_SINGLE_BIRD
    assert resp.verdicts[2].category == curator_pb2.PHOTO_CATEGORY_EGGS


@pytest.mark.parametrize("size", [0, 9])
def test_empty_or_too_big_batch_is_invalid_argument(stub, size):
    client, _ = stub
    photos = [(str(i), webp(400, 300)) for i in range(size)]
    with pytest.raises(grpc.RpcError) as err:
        client.ClassifyPhotos(request(*photos))
    assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_eight_photos_is_accepted(stub):
    client, _ = stub
    resp = client.ClassifyPhotos(request(*[(str(i), webp(400, 300)) for i in range(8)]))
    assert len(resp.verdicts) == 8


def test_health_check_reports_serving(stub):
    _, channel = stub
    health = health_pb2_grpc.HealthStub(channel)
    for service in ("", "passarim.curator.v1.PhotoCuratorService"):
        resp = health.Check(health_pb2.HealthCheckRequest(service=service))
        assert resp.status == health_pb2.HealthCheckResponse.SERVING
