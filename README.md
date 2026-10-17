# passarim-photo-curator

Serviço gRPC, em Python, que classifica fotos de aves para o Passarim: ave sozinha, bando, ninho, ovos, pena, ave
morta, sem ave ou outro. Só o worker do `passarim-catalog` o chama, durante a ingestão.

Contrato: `passarim.curator.v1.PhotoCuratorService`, no [passarim-proto](https://github.com/velosobr/passarim-proto).
Decisão: ADR-0021 no `passarim-docs`.
