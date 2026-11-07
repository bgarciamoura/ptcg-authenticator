import asyncio
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_module
from app.pipeline.storage import InferenceStore


class StubAnalyzer:
    loaded = True
    load_error = None

    def __init__(self, store):
        self.store = store
        self.received = []

    def status(self):
        return {"classifier": False}

    def analyze(self, img, *, raw=None, raw_format=None, sha256=None):
        self.received.append((img.shape, raw_format, sha256))
        return {"verdict": "inconclusiva", "id": "a" * 32}


@pytest.fixture
def client(tmp_path, monkeypatch):
    stub = StubAnalyzer(InferenceStore(tmp_path))
    monkeypatch.setattr(main_module, "analyzer", stub)
    monkeypatch.setattr(main_module.settings, "feedback_token", None)
    # sem `with`: o lifespan (carregamento real) não roda
    return TestClient(main_module.app), stub


def _png_bytes(w=64, h=90) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 50, 50)).save(buf, format="PNG")
    return buf.getvalue()


def test_health_ok_and_loading(client):
    c, stub = client
    r = c.get("/health")
    assert r.status_code == 200 and r.headers["x-request-id"]
    stub.loaded = False
    assert c.get("/health").status_code == 503
    assert c.get("/health").json()["status"] == "loading"
    stub.loaded, stub.load_error = True, "boom"
    assert c.get("/health").json()["status"] == "error"


def test_analyze_upload(client):
    c, stub = client
    r = c.post("/analyze", files={"file": ("c.png", _png_bytes(), "image/png")}, headers={"X-Request-ID": "req-1"})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "inconclusiva" and body["request_id"] == "req-1"
    shape, fmt, sha = stub.received[0]
    assert shape == (90, 64, 3) and fmt == "PNG" and len(sha) == 64


def test_analyze_rejects_bad_input(client):
    c, _ = client
    assert c.post("/analyze", files={"file": ("c.gif", b"GIF89a", "image/gif")}).status_code == 415
    assert c.post("/analyze", files={"file": ("c.png", b"not an image", "image/png")}).status_code == 400


def test_analyze_while_loading(client):
    c, stub = client
    stub.loaded = False
    assert c.post("/analyze", files={"file": ("c.png", _png_bytes(), "image/png")}).status_code == 503


def test_analyze_backpressure(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr(main_module, "_slots", asyncio.Semaphore(0))  # nenhum slot livre
    monkeypatch.setattr(main_module.settings, "queue_timeout_s", 0.05)
    r = c.post("/analyze", files={"file": ("c.png", _png_bytes(), "image/png")})
    assert r.status_code == 503 and "ocupado" in r.json()["detail"]


def test_feedback_flow(client, tmp_path):
    c, stub = client
    inference_id = stub.store.save(np.zeros((88, 63, 3), np.uint8), {"verdict": "inconclusiva"})
    assert c.post("/feedback/not-an-id", json={"label": "real"}).status_code == 400
    assert c.post(f"/feedback/{'f' * 32}", json={"label": "real"}).status_code == 404
    assert c.post(f"/feedback/{inference_id}", json={"label": "maybe"}).status_code == 422
    r = c.post(f"/feedback/{inference_id}", json={"label": "fake", "note": "holo errado"})
    assert r.status_code == 200 and r.json()["label"] == "fake" and r.json()["status"] == "pending"
    assert (tmp_path / inference_id / "label.json").exists()


def test_feedback_token(client, tmp_path, monkeypatch):
    c, stub = client
    monkeypatch.setattr(main_module.settings, "feedback_token", "s3cret")
    inference_id = stub.store.save(np.zeros((88, 63, 3), np.uint8), {})
    assert c.post(f"/feedback/{inference_id}", json={"label": "real"}).status_code == 401
    assert c.post(f"/feedback/{inference_id}", json={"label": "real"}, headers={"X-Feedback-Token": "nope"}).status_code == 401
    assert c.post(f"/feedback/{inference_id}", json={"label": "real"}, headers={"X-Feedback-Token": "s3cret"}).status_code == 200
