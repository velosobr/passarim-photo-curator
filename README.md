# passarim-photo-curator

Serviço gRPC, em Python, que classifica fotos de aves para o Passarim: ave sozinha, bando, ninho, ovos, pena, ave
morta, sem ave ou outro. Só o worker do `passarim-catalog` o chama, durante a ingestão; o Passarim no ar não depende
dele.

- Contrato: `passarim.curator.v1.PhotoCuratorService` no [passarim-proto](https://github.com/velosobr/passarim-proto)
  (v0.3.0).
- Decisão: ADR-0021 no `passarim-docs`. Spec: `docs/superpowers/specs/2026-10-17-passarim-frente-a-curadoria-design.md`.

## Por onde começar a ler

1. `src/photo_curator/rules.py`: a regra de decisão (função pura).
2. `src/photo_curator/server.py`: o servidor gRPC.
3. `src/photo_curator/detector.py` e `zeroshot.py`: os modelos.
4. `tests/test_rules.py`: os casos que a regra cobre.

## Modelos e licenças

| Modelo | Uso | Código | Pesos | Revisão |
|---|---|---|---|---|
| RT-DETR R50vd (`PekingU/rtdetr_r50vd`) | Detector, classe "bird" do COCO | Apache 2.0 | Apache 2.0 | `df939e66…` |
| OpenCLIP ViT-B/32 (`laion/CLIP-ViT-B-32-laion2B-s34B-b79K`) | Classificação sem treino | MIT | MIT | `1a25a446…` |

O Ultralytics (YOLO ou o RT-DETR dele) não é usado por ser AGPL.

**Aviso do card do LAION:** o card dos pesos do CLIP desaconselha uso implantado sem teste no domínio. Aqui o uso é
offline, com classes fixas, sem decisões sobre pessoas, medido num conjunto de referência e com revisão humana da foto
principal de cada espécie.

## Rodando

```bash
uv sync                                   # instala tudo (torch só de CPU)
uv run python -m photo_curator.weights models   # baixa e confere os pesos (~750 MB)
uv run pytest                             # testes rápidos (sem pesos)
CURATOR_MODELS_DIR=models uv run pytest -m models   # testes com os modelos de verdade
MODELS_DIR=models uv run python -m photo_curator     # servidor na porta 50061
```

Imagem Docker:

```bash
docker build -t passarim-photo-curator:dev .
docker run --rm --network none --read-only --tmpfs /tmp -v "$PWD/tests/fixtures:/fixtures:ro" \
  passarim-photo-curator:dev python -m photo_curator.selfcheck /fixtures
```

As fotos de `tests/fixtures` mantêm a licença de cada uma (ver `tests/fixtures/ATTRIBUTION.md`).
