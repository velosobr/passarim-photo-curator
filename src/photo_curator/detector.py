"""O detector de aves: RT-DETR R50vd (pelo Hugging Face transformers).

RT-DETR é um detector de objetos: recebe a foto e devolve caixas com a classe
de cada objeto (entre as 80 do conjunto COCO) e a confiança. Usamos só a
classe "bird". Licença: Apache 2.0 (código e pesos). Ultralytics NÃO: é AGPL.

torch.no_grad(): diz ao torch que não vamos treinar, então ele não guarda as
contas intermediárias (gasta menos memória e é mais rápido).

Quem usa: __main__.py. Cumpre a interface models.Detector.
"""

from pathlib import Path

import torch
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from photo_curator import config
from photo_curator.types import Box

BIRD_LABEL = "bird"


class RtDetrDetector:
    def __init__(self, model_dir: Path):
        # from_pretrained lê de uma PASTA local (os pesos já estão na imagem).
        self._processor = RTDetrImageProcessor.from_pretrained(model_dir)
        self._model = RTDetrForObjectDetection.from_pretrained(model_dir).eval()
        # id2label mapeia número -> nome da classe; achamos o número de "bird"
        # (14 no COCO) em vez de escrever 14 no código.
        labels = self._model.config.id2label
        self._bird_id = next(i for i, name in labels.items() if name == BIRD_LABEL)

    def detect(self, image: Image.Image) -> list[Box]:
        with torch.no_grad():
            inputs = self._processor(images=image, return_tensors="pt")
            outputs = self._model(**inputs)
            # post_process converte a saída crua em caixas na escala da foto
            # original (target_sizes é (altura, largura)) e já corta pela confiança.
            result = self._processor.post_process_object_detection(
                outputs,
                target_sizes=torch.tensor([[image.height, image.width]]),
                threshold=config.DETECTOR_MIN_SCORE,
            )[0]
        return [
            Box(*(float(v) for v in box.tolist()), score=float(score))
            for score, label, box in zip(result["scores"], result["labels"], result["boxes"], strict=True)
            if int(label) == self._bird_id
        ]
