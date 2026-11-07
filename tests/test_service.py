import json

import numpy as np

from app.config import Settings
from app.pipeline.embedder import REFERENCE_GROUPS
from app.pipeline.fusion import SIGNAL_NAMES
from app.pipeline.models import FusionModel, ReferenceAnomaly
from app.pipeline.ocr import OcrLine
from app.pipeline.reference import ReferenceCard
from app.pipeline.registry import new_run_dir, promote
from app.pipeline.service import Analyzer
from app.pipeline.storage import InferenceStore
from tests.stubs import FixedClassifier, StubEmbedder, StubOCR, StubReference
from tests.synthetic import hires_photo, synthetic_photo

REF_CARD = ReferenceCard(
    id="x-1", source="tcgdex", name="Pikachu", hp="60", number="25", printed_total=165, set_name="S",
    attacks=[], image_url="https://assets.tcgdex.net/en/x/1/high.png",
)
OCR_LINES = [OcrLine("Pikachu", 0.9, 0.1, 0.03, 0.4, 0.08), OcrLine("HP60", 0.9, 0.7, 0.03, 0.9, 0.06)]


def _settings(tmp_path, **over) -> Settings:
    base = dict(
        enable_ocr=False, enable_reference=False, store_inferences=False,
        models_dir=tmp_path / "models", data_dir=tmp_path / "data", inferences_dir=tmp_path / "inf",
    )
    base.update(over)
    return Settings(_env_file=None, **base)


def _analyzer(tmp_path, **kw) -> Analyzer:
    a = Analyzer(_settings(tmp_path, **kw.pop("settings", {})), embedder=StubEmbedder(), **kw)
    a.load()
    return a


def test_without_models_is_inconclusive(tmp_path):
    result = _analyzer(tmp_path).analyze(hires_photo())
    assert result["verdict"] == "inconclusiva"
    assert result["card_detected"] and result["fake_probability"] is None
    assert all(v is None for v in result["signals"].values())
    assert "id" not in result  # store desligado
    assert result["px_per_mm"] > 14 and result["texture_reliable"] is True
    assert set(result["timings_ms"]) >= {"rectify", "embed", "total"}


def test_detection_failure_disables_all_signals(tmp_path):
    a = _analyzer(tmp_path)
    a.classifier = FixedClassifier(0.99, a.embedder.groups)
    result = a.analyze(np.full((500, 500, 3), 128, np.uint8))
    assert result["card_detected"] is False
    assert result["signals"]["visual"] is None
    assert result["verdict"] == "inconclusiva"
    assert any("não detectada" in w for w in result["warnings"])


def test_classifier_drives_verdict(tmp_path):
    a = _analyzer(tmp_path)
    a.classifier = FixedClassifier(0.95, a.embedder.groups)
    assert a.analyze(hires_photo())["verdict"] == "falsa"
    a.classifier = FixedClassifier(0.05, a.embedder.groups)
    assert a.analyze(hires_photo())["verdict"] == "real"


def test_low_resolution_blocks_real_verdict(tmp_path):
    a = _analyzer(tmp_path)
    a.classifier = FixedClassifier(0.05, a.embedder.groups)
    result = a.analyze(synthetic_photo())  # 440 px de altura ≈ 5 px/mm
    assert result["texture_reliable"] is False
    assert result["verdict"] == "inconclusiva"
    assert any("resolução baixa" in w for w in result["warnings"])
    a.classifier = FixedClassifier(0.95, a.embedder.groups)
    assert a.analyze(synthetic_photo())["verdict"] == "falsa"  # falsa continua possível


def test_reference_signal_requires_trained_reference_anomaly(tmp_path):
    reference = StubReference(REF_CARD, np.full((880, 630, 3), 200, np.uint8))
    a = _analyzer(tmp_path, ocr=StubOCR(OCR_LINES), reference=reference)
    a.classifier = FixedClassifier(0.2, a.embedder.groups)

    result = a.analyze(hires_photo())
    assert reference.calls == 1
    assert result["reference"]["name"] == "Pikachu" and "similarity" in result["reference"]
    assert result["signals"]["reference"] is None  # sem ReferenceAnomaly: só informativo
    assert result["signals"]["text"] == 0.0  # OCR bateu com a referência
    assert abs(result["fake_probability"] - (0.45 * 0.2) / (0.45 + 0.15)) < 1e-3

    # com ReferenceAnomaly treinado, a distância vira probabilidade
    a.ref_anomaly = ReferenceAnomaly(list(REFERENCE_GROUPS), {}).fit(np.full((6, len(REFERENCE_GROUPS)), 0.01))
    result = a.analyze(hires_photo())
    assert result["signals"]["reference"] is not None
    assert "anomaly_score" in result["reference"]


def test_fusion_rejected_when_components_differ(tmp_path):
    settings = _settings(tmp_path)
    run = new_run_dir(settings.models_dir)
    promote(settings.models_dir, run)
    embedder = StubEmbedder()
    rows = [{"visual": 0.9, "anomaly": None, "text": None, "reference": None}] * 6
    rows += [{"visual": 0.1, "anomaly": None, "text": None, "reference": None}] * 6
    y = np.array([1] * 6 + [0] * 6)
    none = {"classifier": False, "anomaly": False, "ref_anomaly": False, "ocr": False, "reference": False}

    wrong = {"embedder": embedder.meta(), "components": {**none, "ocr": True}}
    FusionModel(SIGNAL_NAMES, wrong).fit(rows, y).save(run / "fusion.json")
    a = Analyzer(settings, embedder=embedder)
    a.load()
    assert a.run_name == run.name and a.fusion_model is None

    FusionModel(SIGNAL_NAMES, {"embedder": embedder.meta(), "components": none}).fit(rows, y).save(run / "fusion.json")
    a = Analyzer(settings, embedder=embedder)
    a.load()
    assert a.fusion_model is not None
    assert a.analyze(hires_photo())["fusion"] == "treinada"


def test_thresholds_from_run_and_hot_reload(tmp_path):
    settings = _settings(tmp_path)
    a = Analyzer(settings, embedder=StubEmbedder())
    a.load()
    assert a.run_name is None
    assert a.status()["thresholds"] == {"threshold_fake": 0.65, "threshold_real": 0.35}

    # um novo run é promovido enquanto a API está de pé
    run = new_run_dir(settings.models_dir)
    (run / "thresholds.json").write_text(json.dumps({"threshold_fake": 0.8, "threshold_real": 0.2}))
    promote(settings.models_dir, run)
    a.classifier = FixedClassifier(0.7, a.embedder.groups)  # 0.7 < 0.8: não é mais "falsa"
    result = a.analyze(hires_photo())
    assert a.run_name == run.name  # recarregou sozinho
    assert result["models"]["thresholds"]["threshold_fake"] == 0.8
    # o reload zera modelos em memória que não estão no run (o FixedClassifier injetado)
    assert result["signals"]["visual"] is None

    (run / "thresholds.json").write_text("{bad json")
    promote(settings.models_dir, run)  # mtime muda
    assert a.maybe_reload() is True and a.thresholds is None


def test_inferences_are_persisted_with_dedup(tmp_path):
    store = InferenceStore(tmp_path / "inf")
    a = _analyzer(tmp_path, store=store)
    img = hires_photo()
    result = a.analyze(img, raw=b"rawbytes", raw_format="PNG", sha256="ab" * 32)
    folder = tmp_path / "inf" / result["id"]
    assert (folder / "card.jpg").exists() and (folder / "original.png").read_bytes() == b"rawbytes"
    saved = json.loads((folder / "result.json").read_text())
    assert saved["verdict"] == "inconclusiva" and "preview" not in saved and saved["quad"] is not None
    assert result["duplicate"] is False

    again = a.analyze(img, raw=b"rawbytes", raw_format="PNG", sha256="ab" * 32)
    assert again["id"] == result["id"] and again["duplicate"] is True
    assert store.stats()["total"] == 1

    store.set_label(result["id"], "fake", "borda errada", source="127.0.0.1")
    assert list(store.iter_labeled()) == []  # pendente não entra no treino
    assert [lbl for _, lbl, _ in store.iter_labeled(include_pending=True)] == [1]
    store.review(result["id"], "approved", "me")
    assert [lbl for _, lbl, _ in store.iter_labeled()] == [1]
    assert store.stats() == {"total": 1, "labeled": 1, "pending": 0, "approved": 1}
