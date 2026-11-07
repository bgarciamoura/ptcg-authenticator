"""Decodificação segura de imagens enviadas pelo usuário."""
from __future__ import annotations

import base64
import hashlib
import io

import cv2
import numpy as np
from PIL import Image, ImageOps

# Proteção contra "decompression bombs". 25 MP é mais que qualquer celular
# comum produz e limita a memória por requisição (~75 MB decodificados).
Image.MAX_IMAGE_PIXELS = 25_000_000

ALLOWED_FORMATS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}


def image_format(data: bytes) -> str:
    """Formato real (pelo conteúdo, não pela extensão): 'JPEG' | 'PNG' | 'WEBP'."""
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            probe.verify()  # checa integridade sem decodificar tudo
    except (OSError, Image.DecompressionBombError, SyntaxError) as exc:
        raise ValueError(f"imagem inválida: {exc}") from exc
    if fmt not in ALLOWED_FORMATS:
        raise ValueError(f"formato não suportado: {fmt}")
    return fmt


def decode_image(data: bytes) -> np.ndarray:
    """Converte bytes em imagem BGR (formato do OpenCV).

    Raises:
        ValueError: se os bytes não forem uma imagem válida/permitida.
    """
    image_format(data)
    try:
        with Image.open(io.BytesIO(data)) as img:
            img = ImageOps.exif_transpose(img)  # respeita rotação do celular
            rgb = np.asarray(img.convert("RGB"))
    except (OSError, Image.DecompressionBombError, SyntaxError) as exc:
        raise ValueError(f"imagem inválida: {exc}") from exc
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def read_image_file(path) -> np.ndarray:
    with open(path, "rb") as fh:
        return decode_image(fh.read())


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_jpeg_base64(img_bgr: np.ndarray, max_side: int = 420) -> str:
    """Miniatura em base64 para exibir na interface."""
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")
