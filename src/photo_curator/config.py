"""Limites da regra de decisão e constantes do serviço.

Os limites abaixo são o ponto de partida. O plano A2 os calibra no conjunto
de referência (metade de calibração) e atualiza este arquivo: por isso eles
ficam todos aqui, num lugar só, e não espalhados pelo código.

Quem usa: rules.py, detector.py, sharpness.py e server.py.
"""

# Versão dos modelos + da regra. Muda quando um modelo, uma frase do CLIP ou um
# limite muda: o worker grava este texto em cada foto e sabe o que refazer.
MODEL_VERSION = "rtdetr-r50vd+openclip-b32@1"

# Lote máximo por chamada (o contrato diz 8: bem abaixo dos 4 MB do gRPC).
MAX_BATCH = 8

# Maior imagem aceita, em pixels. A variante medium tem até 800 px de largura
# (perto de 1 milhão de pixels); 4096 x 4096 dá folga de sobra. Acima disso a
# foto nem é decodificada: um WebP de poucos KB pode descrever uma imagem
# gigante (a "bomba de descompressão") e estourar a memória do container.
MAX_IMAGE_PIXELS = 4096 * 4096

# Detector: só caixas da classe "bird" com confiança a partir daqui contam como ave.
DETECTOR_MIN_SCORE = 0.5

# Descarte por conteúdo: se a probabilidade do CLIP para a classe passa do
# limite, a foto é descartada com a categoria correspondente.
DISCARD_MIN: dict[str, float] = {
    "nest": 0.5,
    "eggs": 0.5,
    "feather": 0.5,
    "dead": 0.5,
    "in_hand": 0.5,
    "drawing": 0.5,
}

# Bando pelo CLIP: a partir daqui é FLOCK, mesmo que o detector ache 0 ou 1 ave.
FLOCK_MIN = 0.6

# Sem ave: com nenhuma detecção, o CLIP precisa dar "no_bird" a partir daqui
# para a foto ser NO_BIRD (senão é OTHER: ave pequena ou escondida).
NO_BIRD_MIN = 0.5

# Detecção falsa: "no_bird" muito forte e a melhor detecção fraca -> NO_BIRD.
NO_BIRD_STRONG = 0.8
WEAK_DETECTION = 0.7

# Dominância: a maior ave com pelo menos DOMINANCE vezes a área da segunda conta
# como "ave sozinha" (cobre o passarinho pequeno no fundo).
DOMINANCE = 4.0

# Borda: a caixa a menos de 2% do lado da foto encosta na borda (ave cortada).
EDGE_TOLERANCE = 0.02

# Nitidez: a variância do Laplaciano (ver sharpness.py) cresce muito de uma foto
# para outra (de 10 a 10 000), então usamos log10 e levamos o intervalo
# [SHARPNESS_LOG_MIN, SHARPNESS_LOG_MAX] para [0, 1].
SHARPNESS_LOG_MIN = 1.0
SHARPNESS_LOG_MAX = 3.5
