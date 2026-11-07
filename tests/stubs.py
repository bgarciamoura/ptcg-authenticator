"""Dublês leves dos componentes pesados, para testar a orquestração sem torch/docTR."""
from __future__ import annotations

import numpy as np

from app.pipeline.embedder import ALL_GROUPS

DIM = 8


class StubEmbedder:
    """Features determinísticas que dependem da COR da carta: cartas claras e
    escuras ficam em regiões distintas do espaço, então um classificador
    consegue separá-las (útil para testar o pipeline de treino)."""

    device = "cpu"
    groups = list(ALL_GROUPS)
    dim = DIM

    def meta(self) -> dict:
        return {"model_name": "stub", "input_size": 224, "groups": list(self.groups)}

    def card_features(self, card_bgr: np.ndarray) -> np.ndarray:
        mean = card_bgr.reshape(-1, 3).mean(axis=0) / 255.0  # (b, g, r)
        rng = np.random.default_rng(int(mean.sum() * 1000) % (2**32))
        base = np.concatenate([mean, 1 - mean, [mean.mean(), 1 - mean.mean()]])  # 8 dims
        feats = base[None, :] + 0.05 * rng.normal(size=(len(self.groups), DIM))
        feats /= np.linalg.norm(feats, axis=1, keepdims=True)
        return feats.reshape(-1).astype(np.float32)


class StubOCR:
    def __init__(self, lines):
        self.lines = lines

    def read_card(self, card_bgr):
        return self.lines


class StubReference:
    def __init__(self, card, image: np.ndarray | None):
        self.card, self.image = card, image
        self.calls = 0

    def find_card(self, name, number, total):
        self.calls += 1
        return self.card

    def get_image(self, card):
        return self.image


class FixedClassifier:
    """Finge um VisualClassifier que devolve sempre a mesma probabilidade."""

    def __init__(self, prob: float, groups):
        self.prob, self.groups, self.meta = prob, list(groups), {}

    def predict_proba(self, X):
        return np.array([self.prob])
