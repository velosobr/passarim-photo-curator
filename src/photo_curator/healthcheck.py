"""Health check para o Docker/compose: `python -m photo_curator.healthcheck`.

Pergunta ao servidor local (grpc.health.v1) se o PhotoCuratorService está
SERVING. Sai com 0 se sim e 1 se não (o Docker marca o container como
"unhealthy"). Usamos Python porque a imagem não tem o grpc_health_probe.
"""

import os
import sys

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

from photo_curator.server import SERVICE_NAME


def main() -> int:
    port = os.environ.get("PORT", "50061")
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            resp = health_pb2_grpc.HealthStub(channel).Check(
                health_pb2.HealthCheckRequest(service=SERVICE_NAME), timeout=3
            )
    except grpc.RpcError:
        return 1
    return 0 if resp.status == health_pb2.HealthCheckResponse.SERVING else 1


if __name__ == "__main__":
    sys.exit(main())
