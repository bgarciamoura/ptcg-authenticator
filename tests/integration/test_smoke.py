"""Smoke tests com os componentes reais (só na imagem runtime): pytest -m integration."""
import numpy as np
import pytest

from app.config import Settings
from app.pipeline.service import Analyzer
from tests.synthetic import hires_photo

pytestmark = pytest.mark.integration
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def embedder():
    from app.pipeline.embedder import Embedder

    return Embedder("facebook/dinov2-small", "cpu", 224)  # local_files_only: pesos embutidos


def test_embedder_shapes(embedder):
    from app.pipeline.detect import CARD_H, CARD_W

    card = np.full((CARD_H, CARD_W, 3), 180, np.uint8)
    feats = embedder.card_features(card)
    assert feats.shape == (len(embedder.groups) * embedder.dim,)
    per_group = feats.reshape(len(embedder.groups), -1)
    assert np.allclose(np.linalg.norm(per_group, axis=1), 1.0, atol=1e-4)


def test_embedder_refuses_unknown_model_without_download():
    from app.pipeline.embedder import Embedder

    with pytest.raises(RuntimeError, match="não está embutido"):
        Embedder("facebook/dinov2-base", "cpu", 224, allow_download=False)


def test_full_pipeline_with_real_ocr(embedder, tmp_path):
    settings = Settings(
        _env_file=None, enable_reference=False, store_inferences=False,
        models_dir=tmp_path, data_dir=tmp_path, inferences_dir=tmp_path,
    )
    analyzer = Analyzer(settings, embedder=embedder)
    analyzer.load()  # inicializa o docTR com os pesos embutidos na imagem
    assert analyzer.ocr is not None, "docTR deveria estar disponível na imagem runtime"
    result = analyzer.analyze(hires_photo(angle=0, text=True))
    assert result["card_detected"] and result["verdict"] == "inconclusiva"
    assert result["texture_reliable"] is True
    ocr = result["ocr"]
    assert ocr["number"] == "025" and ocr["total"] == "165", ocr
    assert ocr["name"] and ocr["name"].lower().startswith("pika"), ocr
    assert result["timings_ms"]["ocr"] > 0
