import io

import pytest
from PIL import Image

from app.pipeline.imageio import decode_image, image_format, sha256_hex


def _bytes(img: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


def test_formats_and_decode():
    img = Image.new("RGB", (40, 30), (255, 0, 0))
    for fmt, expected in (("PNG", "PNG"), ("JPEG", "JPEG"), ("WEBP", "WEBP")):
        data = _bytes(img, fmt)
        assert image_format(data) == expected
        bgr = decode_image(data)
        assert bgr.shape == (30, 40, 3) and bgr[0, 0].tolist()[2] > 200  # canal R em BGR


def test_exif_orientation_is_applied():
    img = Image.new("RGB", (40, 30))
    exif = img.getexif()
    exif[0x0112] = 6  # rotacionar 90° no sentido horário
    data = _bytes(img, "JPEG", exif=exif.tobytes())
    assert decode_image(data).shape[:2] == (40, 30)  # virou retrato


def test_rejects_invalid_and_disallowed():
    with pytest.raises(ValueError):
        decode_image(b"not an image")
    with pytest.raises(ValueError):
        image_format(_bytes(Image.new("P", (4, 4)), "GIF"))


def test_rejects_decompression_bomb():
    # 8000x7000 em modo 1 bit: PNG minúsculo, 56 MP decodificados (> 2x o limite)
    data = _bytes(Image.new("1", (8000, 7000)), "PNG")
    with pytest.raises(ValueError):
        decode_image(data)


def test_sha256():
    assert sha256_hex(b"abc").startswith("ba7816bf")
