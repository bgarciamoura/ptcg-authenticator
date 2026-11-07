import numpy as np

from app.pipeline.embedder import group_distances, group_similarity, select_groups
from app.pipeline.fusion import SIGNAL_NAMES, Signals, anomaly_to_prob, fuse, verdict
from app.pipeline.models import FusionModel, KnnAnomaly, ReferenceAnomaly, VisualClassifier, load_if_exists

META = {"embedder": {"model_name": "stub", "groups": ["a", "b"]}}


def test_fusion_heuristic_and_verdict():
    assert fuse(Signals()) is None
    s = Signals(visual=0.9, anomaly=0.8)
    p = fuse(s)
    assert 0.8 <= p <= 0.9
    assert verdict(p, s, 0.65, 0.35) == "falsa"
    s_real = Signals(visual=0.1, text=0.0)
    assert verdict(fuse(s_real), s_real, 0.65, 0.35) == "real"


def test_reference_counts_as_probability():
    assert fuse(Signals(reference=0.9)) == 0.9
    s = Signals(visual=0.2, reference=0.8)
    assert abs(fuse(s) - (0.45 * 0.2 + 0.20 * 0.8) / 0.65) < 1e-9


def test_real_requires_visual_detection_and_resolution():
    s = Signals(text=0.0)
    assert verdict(fuse(s), s, 0.65, 0.35) == "inconclusiva"
    s = Signals(visual=0.05)
    assert verdict(fuse(s), s, 0.65, 0.35, card_detected=False) == "inconclusiva"
    assert verdict(fuse(s), s, 0.65, 0.35, texture_reliable=False) == "inconclusiva"
    # baixa resolução ainda permite "falsa" com evidência forte
    s = Signals(visual=0.95)
    assert verdict(fuse(s), s, 0.65, 0.35, texture_reliable=False) == "falsa"


def test_anomaly_to_prob_midpoint():
    assert abs(anomaly_to_prob(1.0) - 0.5) < 1e-9
    assert anomaly_to_prob(2.0) > 0.9


def test_select_groups_and_similarity():
    feats = np.arange(12, dtype=float)  # 3 grupos de dim 4
    sel = select_groups(feats, ["a", "b", "c"], ["c", "a"])
    assert sel.tolist() == [8, 9, 10, 11, 0, 1, 2, 3]
    batch = select_groups(np.stack([feats, feats]), ["a", "b", "c"], ("b",))
    assert batch.shape == (2, 4)
    unit = np.tile([1.0, 0, 0, 0], 3)
    assert group_similarity(unit, unit, 3).tolist() == [1.0, 1.0, 1.0]
    assert group_distances(unit, unit, 3).tolist() == [0.0, 0.0, 0.0]


def test_knn_anomaly_roundtrip_and_outlier(tmp_path):
    rng = np.random.default_rng(0)
    center = rng.normal(size=64)
    real = center + 0.05 * rng.normal(size=(30, 64))
    model = KnnAnomaly(["a"], META, k=3).fit(real)
    outlier = rng.normal(size=64)
    assert model.normalized(outlier) > model.normalized(center) * 2
    model.save(tmp_path / "a.npz")
    loaded = KnnAnomaly.load(tmp_path / "a.npz")
    assert loaded.meta == META and loaded.groups == ["a"] and loaded.k == 3
    assert abs(loaded.score(outlier) - model.score(outlier)) < 1e-6


def test_reference_anomaly_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    real = np.abs(rng.normal(0.05, 0.01, size=(20, 3)))
    model = ReferenceAnomaly(["x", "y", "z"], META).fit(real)
    assert model.normalized(np.array([0.05, 0.05, 0.05])) < 1.0
    assert model.normalized(np.array([0.5, 0.5, 0.5])) > 3.0
    model.save(tmp_path / "r.json")
    loaded = ReferenceAnomaly.load(tmp_path / "r.json")
    assert loaded.groups == ["x", "y", "z"] and loaded.n_fit == 20
    assert np.allclose(loaded.thresholds, model.thresholds)


def test_visual_classifier_pca_calibration_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 1, (20, 500)), rng.normal(0.6, 1, (20, 500))])  # 500 dims, 40 amostras
    y = np.array([0] * 20 + [1] * 20)
    clf = VisualClassifier(["a", "b"], META, C=0.1, n_components=8).fit(X, y)
    assert clf.components.shape == (8, 500)
    assert clf.predict_proba(X[:20]).mean() < 0.5 < clf.predict_proba(X[20:]).mean()
    # calibração: logits escalados por a, b
    clf.calibrate(clf.decision(X), y)
    assert clf.calib_a > 0
    clf.save(tmp_path / "c.json")
    loaded = VisualClassifier.load(tmp_path / "c.json")
    assert loaded.groups == ["a", "b"] and loaded.n_components == 8
    assert np.allclose(loaded.predict_proba(X), clf.predict_proba(X))


def test_visual_classifier_handles_tiny_datasets():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(6, 64))
    y = np.array([0, 0, 0, 1, 1, 1])
    clf = VisualClassifier(["a"], META, n_components=32).fit(X, y)
    assert clf.components.shape[0] == 5  # min(32, n-1, d)
    assert clf.predict_proba(X).shape == (6,)


def test_fusion_model_roundtrip(tmp_path):
    rows = [{"visual": 0.9, "anomaly": None, "text": 0.8, "reference": 0.7}] * 10
    rows += [{"visual": 0.1, "anomaly": None, "text": 0.0, "reference": 0.1}] * 10
    y = np.array([1] * 10 + [0] * 10)
    model = FusionModel(SIGNAL_NAMES, META).fit(rows, y)
    assert model.predict_proba(rows[0]) > 0.5 > model.predict_proba(rows[-1])
    assert set(model.coefficients()) == set(SIGNAL_NAMES)
    model.save(tmp_path / "f.json")
    loaded = FusionModel.load(tmp_path / "f.json")
    assert abs(loaded.predict_proba(rows[0]) - model.predict_proba(rows[0])) < 1e-9


def test_load_if_exists_checks_meta(tmp_path):
    rng = np.random.default_rng(2)
    KnnAnomaly(["a"], META, k=2).fit(rng.normal(size=(6, 8))).save(tmp_path / "a.npz")
    assert load_if_exists(KnnAnomaly.load, tmp_path / "a.npz", META) is not None
    assert load_if_exists(KnnAnomaly.load, tmp_path / "a.npz", {"embedder": {"model_name": "other"}}) is None
    assert load_if_exists(KnnAnomaly.load, tmp_path / "missing.npz") is None
    (tmp_path / "bad.npz").write_bytes(b"not an npz")
    assert load_if_exists(KnnAnomaly.load, tmp_path / "bad.npz") is None
