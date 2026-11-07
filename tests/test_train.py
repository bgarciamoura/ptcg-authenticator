"""Pipeline de treino de ponta a ponta com dublês: prepare → train → fit_fusion."""
import json

import cv2
import numpy as np
import pytest

import app.train.fit_fusion as fit_fusion
import app.train.prepare as prepare
import app.train.train as train
from app.config import Settings
from app.pipeline.registry import resolve_latest
from app.pipeline.service import Analyzer
from tests.stubs import StubEmbedder
from tests.synthetic import hires_photo

REAL_COLOR = (220, 200, 60)
FAKE_COLOR = (90, 70, 190)  # falsas "azuladas": separáveis pelo StubEmbedder


def _write_dataset(root, n_real_cards=3, n_fake_cards=3, photos=2, start_angle=0):
    for label, n, color in (("real", n_real_cards, REAL_COLOR), ("fake", n_fake_cards, FAKE_COLOR)):
        for c in range(n):
            folder = root / label / f"card{c}"
            folder.mkdir(parents=True)
            for p in range(photos):
                img = hires_photo(angle=start_angle + 3 * p + c, card_color=color)
                cv2.imwrite(str(folder / f"p{p}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])


@pytest.fixture
def stub_components(monkeypatch):
    monkeypatch.setattr(prepare, "make_embedder", lambda s: StubEmbedder())
    monkeypatch.setattr(
        fit_fusion, "make_analyzer",
        lambda s: Analyzer(s.model_copy(update={"store_inferences": False}), embedder=StubEmbedder()),
    )


def test_prepare_train_fusion_end_to_end(tmp_path, stub_components, monkeypatch):
    raw, models, validation = tmp_path / "raw", tmp_path / "models", tmp_path / "validation"
    _write_dataset(raw)
    _write_dataset(validation, n_real_cards=3, n_fake_cards=3, photos=2, start_angle=1)
    settings = Settings(
        _env_file=None, data_dir=tmp_path, models_dir=models, inferences_dir=tmp_path / "inf",
        enable_ocr=False, enable_reference=False, store_inferences=False,
    )
    for module in (prepare, train, fit_fusion):
        monkeypatch.setattr(module, "get_settings", lambda: settings)

    # 1. prepare
    features = tmp_path / "features.npz"
    prepare.main(["--raw", str(raw), "--out", str(features), "--processed", str(tmp_path / "proc"),
                  "--no-feedback", "--no-reference"])
    data = np.load(features, allow_pickle=False)
    assert data["X"].shape == (12, len(StubEmbedder.groups) * StubEmbedder.dim)
    assert data["y"].tolist() == [1] * 6 + [0] * 6 or sorted(data["y"].tolist()) == [0] * 6 + [1] * 6
    assert np.isnan(data["ref_dist"]).all()  # --no-reference
    assert json.loads(str(data["pairing"]))["pairing_ratio"] == 1.0  # card0..2 em real e fake
    assert (tmp_path / "proc" / "real" / "card0" / "p0.png").exists()

    # 2. train
    train.main(["--features", str(features), "--models", str(models)])
    run = resolve_latest(models)
    assert run is not None
    metrics = json.loads((run / "metrics.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    assert (run / "classifier.json").exists() and (run / "anomaly.npz").exists()
    assert metrics["ref_anomaly"] is None  # sem distâncias à referência
    assert metrics["classifier"]["groups"] == ["header", "text_box", "footer", "corner", "texture"]
    assert metrics["classifier"]["oof_at_0.5"]["roc_auc"] >= 0.9  # cores bem separáveis
    assert metrics["classifier"]["suggested_thresholds"] is None  # só 12 amostras
    assert not (run / "thresholds.json").exists()
    assert manifest["n_real"] == 6 and manifest["pairing"]["pairing_ratio"] == 1.0

    # a API carregaria esse run e classificaria uma carta "falsa" (azulada)
    a = Analyzer(settings, embedder=StubEmbedder())
    a.load()
    assert a.run_name == run.name and a.classifier is not None
    fake_result = a.analyze(hires_photo(angle=2, card_color=FAKE_COLOR))
    real_result = a.analyze(hires_photo(angle=2, card_color=REAL_COLOR))
    assert fake_result["signals"]["visual"] > real_result["signals"]["visual"]

    # 3. fit_fusion
    fit_fusion.main(["--validation", str(validation)])
    assert (run / "fusion.json").exists()
    a = Analyzer(settings, embedder=StubEmbedder())
    a.load()
    assert a.fusion_model is not None


def test_train_without_fakes_only_trains_anomaly(tmp_path, stub_components, monkeypatch):
    raw, models = tmp_path / "raw", tmp_path / "models"
    _write_dataset(raw, n_real_cards=3, n_fake_cards=0, photos=2)
    settings = Settings(_env_file=None, data_dir=tmp_path, models_dir=models, inferences_dir=tmp_path / "inf")
    for module in (prepare, train):
        monkeypatch.setattr(module, "get_settings", lambda: settings)
    features = tmp_path / "features.npz"
    prepare.main(["--raw", str(raw), "--out", str(features), "--processed", str(tmp_path / "proc"),
                  "--no-feedback", "--no-reference"])
    train.main(["--features", str(features), "--models", str(models), "--no-promote"])
    assert resolve_latest(models) is None  # --no-promote
    runs = list((models / "runs").iterdir())
    assert len(runs) == 1
    metrics = json.loads((runs[0] / "metrics.json").read_text())
    assert metrics["classifier"] is None and metrics["anomaly"]["n_real"] == 6


def test_suggest_thresholds_needs_enough_data():
    rng = np.random.default_rng(0)
    y = np.array([0] * 30 + [1] * 30)
    prob = np.concatenate([rng.uniform(0, 0.3, 30), rng.uniform(0.7, 1.0, 30)])
    t = train.suggest_thresholds(y, prob, (0.65, 0.35))
    assert t is not None and t["threshold_real"] < t["threshold_fake"]
    assert train.suggest_thresholds(y[:20], prob[:20], (0.65, 0.35)) is None
