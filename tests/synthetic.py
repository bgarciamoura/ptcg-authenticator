"""Geradores de fotos sintéticas de carta para os testes."""
from __future__ import annotations

import cv2
import numpy as np

CARD_COLOR = (220, 200, 60)  # BGR: amarelo-ish, como a borda de uma carta Pokémon


def synthetic_photo(
    angle: float = 12.0,
    landscape: bool = False,
    size: int = 1200,
    card_height: int = 440,
    background=(30, 30, 30),
    card_color=CARD_COLOR,
    noise: float = 0.0,
    text: bool = False,
) -> np.ndarray:
    """Fundo uniforme (ou ruidoso) com um retângulo na proporção de carta."""
    img = np.full((size, size, 3), background, np.uint8)
    if noise:
        rng = np.random.default_rng(0)
        img = np.clip(img.astype(float) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    h = card_height
    w = int(round(h * 63 / 88))
    dims = (h, w) if landscape else (w, h)
    box = cv2.boxPoints(((size / 2, size / 2), dims, angle)).astype(np.int32)
    cv2.fillConvexPoly(img, box, card_color)
    if text and not landscape and angle == 0:
        cx, cy = size // 2, size // 2
        s = h / 440
        cv2.putText(img, "Pikachu", (int(cx - w * 0.42), int(cy - h * 0.40)), cv2.FONT_HERSHEY_SIMPLEX, 1.4 * s, (20, 20, 20), max(2, int(4 * s)))
        cv2.putText(img, "HP60", (int(cx + w * 0.15), int(cy - h * 0.40)), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * s, (20, 20, 20), max(2, int(3 * s)))
        cv2.putText(img, "025/165", (int(cx - w * 0.42), int(cy + h * 0.46)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, (20, 20, 20), max(1, int(2 * s)))
    return img


def hires_photo(**kw) -> np.ndarray:
    """Carta com ~15 px/mm (1320 px de altura): textura 'confiável'."""
    kw.setdefault("size", 2000)
    kw.setdefault("card_height", 1320)
    return synthetic_photo(**kw)
