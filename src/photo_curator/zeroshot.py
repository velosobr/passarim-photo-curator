"""A classificação sem treino: OpenCLIP ViT-B/32 com pesos LAION-2B.

O CLIP transforma a foto e cada frase de prompts.py em vetores ("embeddings").
Normalizamos os vetores (tamanho 1) e o produto escalar entre eles vira a
"semelhança" (cosseno). Multiplicamos por 100 (a escala com que o CLIP foi
treinado) e aplicamos softmax para virar probabilidades.

Os vetores das frases não mudam: calculamos uma vez, no __init__.
Licença: MIT (código e pesos). O card do LAION desaconselha uso implantado sem
teste no domínio; o README explica por que o nosso uso é aceitável.

Quem usa: __main__.py. Cumpre a interface models.ZeroShot.
"""

from pathlib import Path

import open_clip
import torch
from PIL import Image

from photo_curator.prompts import aggregate, flat_prompts

ARCHITECTURE = "ViT-B-32"
_LOGIT_SCALE = 100.0


class ClipZeroShot:
    def __init__(self, weights_file: Path):
        # pretrained aceita o caminho de um arquivo local: nada é baixado.
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            ARCHITECTURE, pretrained=str(weights_file)
        )
        self._model.eval()
        tokenizer = open_clip.get_tokenizer(ARCHITECTURE)
        with torch.no_grad():
            text = self._model.encode_text(tokenizer(flat_prompts()))
            self._text = text / text.norm(dim=-1, keepdim=True)

    def classify(self, image: Image.Image) -> dict[str, float]:
        with torch.no_grad():
            img = self._model.encode_image(self._preprocess(image).unsqueeze(0))
            img = img / img.norm(dim=-1, keepdim=True)
            probs = (_LOGIT_SCALE * img @ self._text.T).softmax(dim=-1)[0]
        return aggregate(probs.tolist())
