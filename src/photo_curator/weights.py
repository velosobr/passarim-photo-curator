"""Baixa e confere os pesos dos modelos.

Pesos são os números que o modelo aprendeu no treino (centenas de MB). Eles
ficam DENTRO da imagem Docker: o Dockerfile roda `python -m photo_curator.weights
/opt/models` na hora do build, e em execução o serviço não acessa a internet.

Duas travas garantem que sempre usamos exatamente os mesmos pesos:
- revision: o hash do commit no Hugging Face (nunca "main", que muda);
- sha256: o hash de cada arquivo, conferido depois do download.

Quem usa: o Dockerfile, a CI (job dos modelos) e __main__.py (verify ao subir).
"""

import hashlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

RTDETR_SUBDIR = "rtdetr"
CLIP_FILE = "clip/open_clip_model.safetensors"

_RTDETR = ("PekingU/rtdetr_r50vd", "df939e661d8c52e80608d1ec566561aabd25a4e7")
_CLIP = ("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", "1a25a446712ba5ee05982a381eed697ef9b435cf")


@dataclass(frozen=True)
class WeightFile:
    repo_id: str  # repositório no Hugging Face
    revision: str  # commit fixo
    filename: str  # arquivo dentro do repositório
    subdir: str  # pasta de destino dentro de dest
    sha256: str  # hash esperado do arquivo


FILES: tuple[WeightFile, ...] = (
    WeightFile(
        *_RTDETR,
        "model.safetensors",
        RTDETR_SUBDIR,
        "5263d5521eff3e356f6cd8a371fd5dfb891725beda5f713674f79669115cdc64",
    ),
    WeightFile(
        *_RTDETR,
        "config.json",
        RTDETR_SUBDIR,
        "2ed2a305c51eef46715eb755a02b2a266ecfb752936cc9574bb5714601c2742d",
    ),
    WeightFile(
        *_RTDETR,
        "preprocessor_config.json",
        RTDETR_SUBDIR,
        "ffb4b9461a1dad746be8f0f9c8330ed7743a1ba5fba4f75c232cd281b3d4c64a",
    ),
    WeightFile(
        *_CLIP,
        "open_clip_model.safetensors",
        "clip",
        "ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6",
    ),
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):  # lê de 1 MB em 1 MB
            h.update(chunk)
    return h.hexdigest()


def verify(dest: Path) -> None:
    """Confere que todos os arquivos existem e têm o hash esperado."""
    for f in FILES:
        path = dest / f.subdir / f.filename
        if not path.is_file():
            raise RuntimeError(f"falta o arquivo de pesos {path}")
        got = _sha256(path)
        if got != f.sha256:
            raise RuntimeError(f"sha256 de {path} é {got}, esperado {f.sha256}")


def download(dest: Path) -> None:
    """Baixa cada arquivo na revisão fixa para dest/<subdir>/<filename> e confere."""
    # Import aqui dentro: o serviço em execução não precisa do huggingface_hub
    # para nada além deste download (e roda offline).
    from huggingface_hub import hf_hub_download

    for f in FILES:
        target_dir = dest / f.subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(f.repo_id, f.filename, revision=f.revision)
        shutil.copyfile(cached, target_dir / f.filename)
    verify(dest)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("uso: python -m photo_curator.weights <pasta>")
    download(Path(sys.argv[1]))
    print(f"pesos baixados e conferidos em {sys.argv[1]}")
