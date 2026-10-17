# Imagem do curador em dois estágios ("multi-stage"):
# 1) "build": instala as dependências com o uv e baixa os pesos (precisa de
#    internet e do git, para o pacote do contrato que vem do GitHub);
# 2) final: só o Python, o ambiente pronto e os pesos. Sem uv, sem git, sem
#    compilador: menos coisa para ter vulnerabilidade.

FROM python:3.13-slim AS build
# O uv vem da imagem oficial, numa versão fixa.
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# UV_COMPILE_BYTECODE: já gera os .pyc (a imagem final é só leitura).
# UV_LINK_MODE=copy: copia os pacotes para a .venv em vez de criar links para o cache.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
# Primeiro só os arquivos de dependência: enquanto eles não mudam, o Docker
# reaproveita esta camada (a mais demorada) entre builds.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev
# Pesos dentro da imagem, na revisão e com o hash conferidos (weights.py).
RUN /app/.venv/bin/python -m photo_curator.weights /opt/models

FROM python:3.13-slim
WORKDIR /app
COPY --from=build /app /app
COPY --from=build /opt/models /opt/models
# Usuário sem privilégios: se alguém explorar uma falha, não vira root.
RUN useradd --system --uid 10001 --no-create-home curator
USER 10001
# Em execução, nada de internet: o Hugging Face fica proibido de baixar.
# HOME=/tmp: o usuário não tem pasta própria e o sistema de arquivos é só de
# leitura; bibliotecas que gravam cache em ~ (torch, matplotlib) usam o tmpfs.
ENV PATH="/app/.venv/bin:$PATH" \
    MODELS_DIR=/opt/models \
    PORT=50061 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp
EXPOSE 50061
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=3 CMD ["python", "-m", "photo_curator.healthcheck"]
CMD ["python", "-m", "photo_curator"]
