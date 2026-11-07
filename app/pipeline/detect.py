"""Detecção da carta na foto, retificação e definição das regiões/tiles."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

# Proporção oficial da carta: 63 x 88 mm. 1260x1760 ≈ 20 px/mm — resolução
# suficiente para os tiles de textura verem o padrão de impressão, DESDE QUE
# a foto original tenha essa resolução (ver Rectification.px_per_mm).
CARD_W, CARD_H = 1260, 1760
CARD_MM_H = 88.0
CARD_RATIO = CARD_W / CARD_H  # ~0.716

# Regiões (x0, y0, x1, y1) em fração da carta retificada. Cada região vira um
# embedding separado (com letterbox, sem distorcer a proporção).
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "full": (0.00, 0.00, 1.00, 1.00),
    "header": (0.00, 0.00, 1.00, 0.13),
    "art": (0.07, 0.10, 0.93, 0.50),
    "text_box": (0.00, 0.50, 1.00, 0.90),
    "footer": (0.00, 0.88, 1.00, 1.00),
    "corner": (0.00, 0.00, 0.30, 0.22),
}

# Regiões que NÃO dependem da arte da carta. Classificador e anomalia usam só
# estas (+ textura): a arte identifica QUAL carta é, não SE é falsa.
INVARIANT_REGIONS: tuple[str, ...] = ("header", "text_box", "footer", "corner")

# Tiles de textura: recortes em resolução NATIVA (sem resize) em zonas sem
# arte. (nome, x, y) = canto superior esquerdo em fração da carta.
TEXTURE_TILES: tuple[tuple[str, float, float], ...] = (
    ("tile_top", 0.40, 0.00),
    ("tile_left", 0.00, 0.45),
    ("tile_right", 1.00, 0.45),
    ("tile_text_a", 0.06, 0.56),
    ("tile_text_b", 0.55, 0.70),
    ("tile_footer", 0.00, 1.00),
)

_MAX_DETECT_SIDE = 1024
_MIN_AREA_FRACTION = 0.05  # carta precisa ocupar >= 5% do quadro
_MAX_AREA_FRACTION = 0.98  # ... e não pode ser a moldura da própria foto


@dataclass
class Rectification:
    card: np.ndarray  # CARD_H x CARD_W, retrato
    detected: bool
    quad: np.ndarray | None  # 4x2 na foto original (tl, tr, br, bl)
    source_long_side: float | None  # altura da carta NA FOTO, em pixels

    @property
    def px_per_mm(self) -> float | None:
        """Resolução efetiva da carta na foto original."""
        if not self.detected or not self.source_long_side:
            return None
        return self.source_long_side / CARD_MM_H


def order_points(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 pontos como [topo-esq, topo-dir, base-dir, base-esq]."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def _plausible_card(quad: np.ndarray) -> bool:
    """Descarta quadriláteros com proporção muito diferente de uma carta."""
    tl, tr, br, bl = order_points(quad)
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if min(w, h) < 1:
        return False
    ratio = min(w, h) / max(w, h)
    return abs(ratio - CARD_RATIO) < 0.15


def _edge_maps(small: np.ndarray) -> Iterator[tuple[str, np.ndarray]]:
    """Várias estratégias: cada uma funciona em um tipo de fundo.

    - gray: carta clara sobre fundo escuro (o caso fácil);
    - saturation: borda amarela/prata sobre mesa branca ou clara, onde o
      cinza quase não tem contraste mas a saturação tem;
    - adaptive: fundos com textura (madeira, tecido) e iluminação desigual.
    """
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    yield "gray", cv2.Canny(gray, 50, 150)

    sat = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1], (5, 5), 0)
    yield "saturation", cv2.Canny(sat, 40, 120)

    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
    yield "adaptive", cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def _best_quad(edges: np.ndarray, min_area: float, max_area: float) -> np.ndarray | None:
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(contour)
        if area < min_area:
            break
        if area > max_area:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype(np.float32)
        else:
            # cantos arredondados/oclusão: usa o retângulo mínimo
            quad = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
        if _plausible_card(quad):
            return quad
    return None


def find_card_quad(img_bgr: np.ndarray) -> np.ndarray | None:
    """Retorna os 4 cantos da carta (coordenadas da imagem original) ou None."""
    h, w = img_bgr.shape[:2]
    scale = min(1.0, _MAX_DETECT_SIDE / max(h, w))
    small = (
        cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else img_bgr
    )
    area = small.shape[0] * small.shape[1]
    candidates = []
    for _, edges in _edge_maps(small):
        quad = _best_quad(edges, _MIN_AREA_FRACTION * area, _MAX_AREA_FRACTION * area)
        if quad is not None:
            candidates.append(quad)
    if not candidates:
        return None
    # entre as estratégias que acharam algo plausível, fica a maior carta
    best = max(candidates, key=lambda q: cv2.contourArea(q.reshape(-1, 1, 2)))
    return best / scale


def rectify(img_bgr: np.ndarray) -> Rectification:
    """Recorta e "desentorta" a carta para CARD_W x CARD_H em retrato.

    Se não detectar, apenas redimensiona a foto inteira; quem chama DEVE tratar
    ``detected=False`` como inconclusivo: as regiões não são confiáveis.
    """
    quad = find_card_quad(img_bgr)
    if quad is None:
        card = cv2.resize(img_bgr, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)
        return Rectification(card, False, None, None)

    tl, tr, br, bl = order_points(quad)
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    if width > height:  # carta deitada: gira 90° para ficar em retrato
        src = np.array([bl, tl, tr, br], dtype=np.float32)

    dst = np.array(
        [[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_bgr, matrix, (CARD_W, CARD_H), flags=cv2.INTER_CUBIC)
    return Rectification(warped, True, src, float(max(width, height)))


def crop_region(card_bgr: np.ndarray, region: str) -> np.ndarray:
    x0, y0, x1, y1 = REGIONS[region]
    h, w = card_bgr.shape[:2]
    return card_bgr[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]


def crop_tiles(card_bgr: np.ndarray, size: int) -> list[np.ndarray]:
    """Recorta os tiles de textura em resolução nativa (size x size pixels)."""
    h, w = card_bgr.shape[:2]
    size = min(size, h, w)
    tiles = []
    for _, xf, yf in TEXTURE_TILES:
        x = min(int(xf * w), w - size)
        y = min(int(yf * h), h - size)
        tiles.append(card_bgr[y : y + size, x : x + size])
    return tiles
