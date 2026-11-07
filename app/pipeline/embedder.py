"""Extração de features visuais com DINOv2 (backbone congelado).

O vetor de uma carta é a concatenação de "grupos", todos com a mesma dimensão:
  - um embedding por região de REGIONS (recorte com letterbox, sem distorcer);
  - um embedding de TEXTURA: média dos tiles em resolução nativa.
Cada modelo consome um subconjunto de grupos (``select_groups``):
"""
from __future__ import annotations

import cv2
import numpy as np

from app.pipeline.detect import (
    CARD_H,
    CARD_W,
    INVARIANT_REGIONS,
    REGIONS,
    TEXTURE_TILES,
    crop_region,
    crop_tiles,
)

TEXTURE_GROUP = "texture"
ALL_GROUPS: tuple[str, ...] = (*REGIONS, TEXTURE_GROUP)
# Classificador e anomalia: sem `art`/`full`, senão aprendem a identidade da carta.
CLASSIFIER_GROUPS: tuple[str, ...] = (*INVARIANT_REGIONS, TEXTURE_GROUP)
ANOMALY_GROUPS: tuple[str, ...] = CLASSIFIER_GROUPS
# Comparação com a imagem oficial: só regiões semânticas. A imagem oficial é um
# scan de ~700 px: a textura dela ampliada é ruído, não evidência.
REFERENCE_GROUPS: tuple[str, ...] = tuple(REGIONS)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_PAD_COLOR = 114  # cinza neutro para o letterbox


def letterbox(img_bgr: np.ndarray, size: int) -> np.ndarray:
    """Redimensiona mantendo a proporção e preenche até size x size."""
    h, w = img_bgr.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=interp)
    canvas = np.full((size, size, 3), _PAD_COLOR, dtype=np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


class Embedder:
    def __init__(self, model_name: str, device: str = "auto", input_size: int = 224, allow_download: bool = False):
        import torch  # import tardio: testes leves não precisam de torch
        from transformers import AutoModel

        if input_size % 14:
            raise ValueError("input_size deve ser múltiplo de 14 (patch do DINOv2)")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            model = AutoModel.from_pretrained(model_name, local_files_only=not allow_download)
        except OSError as exc:
            raise RuntimeError(
                f"modelo '{model_name}' não está embutido na imagem. Rebuild com "
                f"DINO_MODEL={model_name} ou defina ALLOW_MODEL_DOWNLOAD=true."
            ) from exc

        self._torch = torch
        self.device = device
        self.model_name = model_name
        self.input_size = input_size
        self.model = model.to(device).eval()
        self.dim = int(self.model.config.hidden_size) * 2  # [CLS ; média dos patches]
        self.groups: list[str] = list(ALL_GROUPS)

    def _to_tensor(self, images_bgr: list[np.ndarray]):
        batch = []
        for img in images_bgr:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            batch.append((rgb.astype(np.float32) / 255.0 - _MEAN) / _STD)
        arr = np.stack(batch).transpose(0, 3, 1, 2)
        return self._torch.from_numpy(np.ascontiguousarray(arr)).to(self.device)

    def embed(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        """Embeddings L2-normalizados (n, dim) de imagens já em input_size."""
        torch = self._torch
        with torch.inference_mode():
            hidden = self.model(pixel_values=self._to_tensor(images_bgr)).last_hidden_state
            feats = torch.cat([hidden[:, 0], hidden[:, 1:].mean(dim=1)], dim=-1)
            feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.float().cpu().numpy()

    def card_features(self, card_bgr: np.ndarray) -> np.ndarray:
        """Vetor 1D (len(groups) * dim) de uma carta retificada."""
        s = self.input_size
        region_imgs = [letterbox(crop_region(card_bgr, r), s) for r in REGIONS]
        tiles = [t if t.shape[:2] == (s, s) else cv2.resize(t, (s, s)) for t in crop_tiles(card_bgr, s)]
        embs = self.embed(region_imgs + tiles)
        n_reg = len(REGIONS)
        texture = _l2(embs[n_reg:].mean(axis=0))
        return np.concatenate([embs[:n_reg].reshape(-1), texture]).astype(np.float32)

    def meta(self) -> dict:
        """Identifica o espaço de features; modelos só valem para o mesmo meta."""
        return {
            "model_name": self.model_name,
            "input_size": self.input_size,
            "card_size": [CARD_W, CARD_H],
            "groups": list(self.groups),
            "tiles": [t[0] for t in TEXTURE_TILES],
        }


def select_groups(feats: np.ndarray, all_groups: list[str], wanted) -> np.ndarray:
    """Extrai os grupos ``wanted`` de vetores produzidos com ``all_groups``."""
    feats = np.asarray(feats)
    dim = feats.shape[-1] // len(all_groups)
    idx = [list(all_groups).index(g) for g in wanted]
    parts = feats.reshape(*feats.shape[:-1], len(all_groups), dim)[..., idx, :]
    return parts.reshape(*feats.shape[:-1], -1)


def group_similarity(feat_a: np.ndarray, feat_b: np.ndarray, n_groups: int) -> np.ndarray:
    """Similaridade de cosseno por grupo entre dois vetores com n_groups grupos."""
    a = feat_a.reshape(n_groups, -1)
    b = feat_b.reshape(n_groups, -1)
    return np.sum(a * b, axis=1)  # já normalizados em L2


def group_distances(feat_a: np.ndarray, feat_b: np.ndarray, n_groups: int) -> np.ndarray:
    """Distância de cosseno (1 - similaridade) por grupo, em [0, 2]."""
    return 1.0 - group_similarity(feat_a, feat_b, n_groups)
