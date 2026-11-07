import numpy as np

from app.pipeline.detect import (
    CARD_H,
    CARD_W,
    REGIONS,
    TEXTURE_TILES,
    crop_region,
    crop_tiles,
    order_points,
    rectify,
)
from tests.synthetic import hires_photo, synthetic_photo


def _center_mean(card):
    return card[CARD_H // 4 : 3 * CARD_H // 4, CARD_W // 4 : 3 * CARD_W // 4].mean()


def test_order_points():
    pts = np.array([[10, 10], [0, 10], [10, 0], [0, 0]], np.float32)
    assert order_points(pts).tolist() == [[0, 0], [10, 0], [10, 10], [0, 10]]


def test_rectify_dark_background():
    rect = rectify(synthetic_photo())
    assert rect.detected and rect.card.shape == (CARD_H, CARD_W, 3)
    assert _center_mean(rect.card) > 120
    assert rect.quad.shape == (4, 2)
    assert 4.5 < rect.px_per_mm < 5.5  # 440 px / 88 mm


def test_rectify_yellow_card_on_white_table():
    """Borda amarela sobre fundo branco: pouco contraste em cinza, muito em saturação."""
    rect = rectify(synthetic_photo(angle=8, background=(245, 245, 245)))
    assert rect.detected
    assert _center_mean(rect.card) < 200  # é a carta (amarela), não o fundo branco


def test_rectify_noisy_background():
    rect = rectify(synthetic_photo(angle=5, background=(120, 100, 80), noise=25))
    assert rect.detected and rect.px_per_mm > 4


def test_rectify_landscape_becomes_portrait():
    rect = rectify(synthetic_photo(angle=5, landscape=True))
    assert rect.detected and rect.card.shape == (CARD_H, CARD_W, 3)


def test_rectify_hires_resolution():
    assert rectify(hires_photo()).px_per_mm > 14


def test_rectify_fallback_without_card():
    rect = rectify(np.full((500, 500, 3), 128, np.uint8))
    assert not rect.detected and rect.card.shape == (CARD_H, CARD_W, 3)
    assert rect.quad is None and rect.px_per_mm is None


def test_rectify_does_not_take_whole_frame():
    """Uma foto 3:4 sem carta tem proporção parecida com a de uma carta: a moldura
    da própria imagem não pode ser aceita como 'carta'."""
    rect = rectify(np.full((800, 600, 3), 200, np.uint8))
    assert not rect.detected


def test_crop_region_and_tiles():
    card = np.zeros((CARD_H, CARD_W, 3), np.uint8)
    assert crop_region(card, "footer").shape[0] == CARD_H - int(0.88 * CARD_H)
    assert set(REGIONS) >= {"full", "art", "footer"}
    tiles = crop_tiles(card, 224)
    assert len(tiles) == len(TEXTURE_TILES)
    assert all(t.shape == (224, 224, 3) for t in tiles)  # resolução nativa, sem resize
