"""Orquestra o pipeline completo de análise de uma foto.

Concorrência: embedder e OCR têm locks separados (nenhum é thread-safe);
chamadas de rede à base oficial ficam FORA de qualquer lock. Os modelos do
run atual são recarregados automaticamente quando models/latest.json muda.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from app.config import Settings
from app.pipeline.detect import CARD_H, CARD_W, rectify
from app.pipeline.embedder import REFERENCE_GROUPS, group_distances, group_similarity, select_groups
from app.pipeline.fusion import Signals, anomaly_to_prob, fuse, verdict
from app.pipeline.imageio import encode_jpeg_base64
from app.pipeline.models import FusionModel, KnnAnomaly, ReferenceAnomaly, VisualClassifier, load_if_exists
from app.pipeline.ocr import ParsedCard, parse_card_text
from app.pipeline.registry import LATEST_FILE, resolve_latest

log = logging.getLogger(__name__)


class _Timer:
    def __init__(self):
        self.ms: dict[str, int] = {}
        self._t = time.perf_counter()

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.ms[name] = self.ms.get(name, 0) + int((now - self._t) * 1000)
        self._t = now


class Analyzer:
    def __init__(self, settings: Settings, *, embedder=None, ocr=None, reference=None, store=None):
        self.s = settings
        self._embed_lock = threading.Lock()
        self._ocr_lock = threading.Lock()
        self._reload_lock = threading.Lock()
        self.loaded = False
        self.load_error: str | None = None
        # componentes injetáveis (testes passam stubs)
        self.embedder = embedder
        self.ocr = ocr
        self.reference = reference
        self.store = store
        self.classifier: VisualClassifier | None = None
        self.anomaly: KnnAnomaly | None = None
        self.ref_anomaly: ReferenceAnomaly | None = None
        self.fusion_model: FusionModel | None = None
        self.thresholds: dict | None = None
        self.run_name: str | None = None
        self._latest_mtime: float | None = None

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        if self.embedder is None:
            from app.pipeline.embedder import Embedder

            log.info("carregando embedder %s ...", self.s.dino_model)
            self.embedder = Embedder(self.s.dino_model, self.s.device, self.s.embed_size, self.s.allow_model_download)

        if self.s.enable_ocr and self.ocr is None:
            from app.pipeline.ocr import CardOCR

            try:
                self.ocr = CardOCR(self.s.allow_model_download)
            except Exception:
                log.exception("OCR indisponível; seguindo sem ele")
        if self.s.enable_reference and self.reference is None:
            from app.pipeline.reference import ReferenceClient

            self.reference = ReferenceClient(
                self.s.reference_cache_dir,
                self.s.tcg_api_url,
                self.s.tcg_api_key,
                self.s.tcgdex_api_url if self.s.enable_tcgdex else None,
                self.s.ocr_lang,
                self.s.reference_timeout,
                self.s.reference_cache_ttl_hours,
            )
        if self.s.store_inferences and self.store is None:
            from app.pipeline.storage import InferenceStore

            self.store = InferenceStore(self.s.inferences_dir)
            try:
                log.info("limpeza de inferências: %s", self.store.cleanup(self.s.retention_days, self.s.max_inferences_gb))
            except OSError:
                log.exception("falha na limpeza de inferências")

        self._load_run()
        self.loaded = True
        log.info("pronto: %s", self.status())

    def _load_run(self) -> None:
        """(Re)carrega os modelos do run apontado por latest.json."""
        emb_meta = self.embedder.meta()
        run_dir = resolve_latest(self.s.models_dir)
        self.classifier = self.anomaly = self.ref_anomaly = self.fusion_model = None
        self.thresholds = None
        self.run_name = None
        self._latest_mtime = self._pointer_mtime()
        if run_dir is None:
            return
        self.run_name = run_dir.name
        expected = {"embedder": emb_meta}
        self.classifier = load_if_exists(VisualClassifier.load, run_dir / "classifier.json", expected)
        self.anomaly = load_if_exists(KnnAnomaly.load, run_dir / "anomaly.npz", expected)
        self.ref_anomaly = load_if_exists(ReferenceAnomaly.load, run_dir / "ref_anomaly.json", expected)
        self.thresholds = self._load_thresholds(run_dir / "thresholds.json")
        # a fusão só vale para a mesma combinação de componentes com que foi ajustada
        self.fusion_model = load_if_exists(
            FusionModel.load, run_dir / "fusion.json", {"embedder": emb_meta, "components": self.components()}
        )

    @staticmethod
    def _load_thresholds(path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            obj = json.loads(path.read_text())
            fake, real = float(obj["threshold_fake"]), float(obj["threshold_real"])
            if 0 < real < fake < 1:
                return {"threshold_fake": fake, "threshold_real": real}
        except (ValueError, KeyError, TypeError):
            log.warning("thresholds.json inválido; usando os padrões")
        return None

    def _pointer_mtime(self) -> float | None:
        try:
            return (self.s.models_dir / LATEST_FILE).stat().st_mtime
        except OSError:
            return None

    def maybe_reload(self) -> bool:
        """Recarrega os modelos se latest.json mudou desde o último load."""
        if self._pointer_mtime() == self._latest_mtime:
            return False
        with self._reload_lock:
            if self._pointer_mtime() == self._latest_mtime:
                return False
            log.info("latest.json mudou; recarregando modelos")
            self._load_run()
            return True

    def components(self) -> dict[str, bool]:
        return {
            "classifier": self.classifier is not None,
            "anomaly": self.anomaly is not None,
            "ref_anomaly": self.ref_anomaly is not None,
            "ocr": self.ocr is not None,
            "reference": self.reference is not None,
        }

    def status(self) -> dict:
        return {
            **self.components(),
            "embedder": self.embedder is not None,
            "device": getattr(self.embedder, "device", None),
            "fusion": self.fusion_model is not None,
            "run": self.run_name,
            "thresholds": self.thresholds or {"threshold_fake": self.s.threshold_fake, "threshold_real": self.s.threshold_real},
            "store": self.store is not None,
        }

    # --------------------------------------------------------------- helpers
    def _embed(self, card_bgr: np.ndarray) -> np.ndarray:
        with self._embed_lock:
            return self.embedder.card_features(card_bgr)

    def _features_for(self, model, feats: np.ndarray) -> np.ndarray:
        return select_groups(feats, self.embedder.groups, model.groups)

    # ---------------------------------------------------------------- signals
    def compute_signals(self, img_bgr: np.ndarray) -> tuple[Signals, dict, np.ndarray]:
        assert self.embedder is not None, "chame load() antes"
        timer = _Timer()
        rect = rectify(img_bgr)
        card = rect.card
        timer.lap("rectify")
        signals = Signals()
        px_per_mm = rect.px_per_mm
        texture_reliable = px_per_mm is not None and px_per_mm >= self.s.min_px_per_mm
        report: dict = {
            "card_detected": rect.detected,
            "quad": None if rect.quad is None else np.round(rect.quad, 1).tolist(),
            "px_per_mm": None if px_per_mm is None else round(px_per_mm, 2),
            "texture_reliable": texture_reliable,
            "warnings": [],
            "text_issues": [],
            "ocr": ParsedCard().to_dict(),
            "timings_ms": timer.ms,
        }

        # Sem carta detectada, regiões e OCR não são confiáveis: nenhum sinal.
        if not rect.detected:
            report["warnings"].append(
                "carta não detectada: enquadre a carta inteira sobre um fundo liso e contrastante"
            )
            report["preview"] = encode_jpeg_base64(card)
            return signals, report, card
        if not texture_reliable:
            report["warnings"].append(
                f"resolução baixa ({px_per_mm:.1f} px/mm; mínimo {self.s.min_px_per_mm:g}): a textura de "
                "impressão não é visível, então a carta não pode ser dada como real. Aproxime a câmera."
            )

        feats = self._embed(card)
        timer.lap("embed")
        if self.classifier is not None:
            signals.visual = float(self.classifier.predict_proba(self._features_for(self.classifier, feats))[0])
        if self.anomaly is not None:
            norm = self.anomaly.normalized(self._features_for(self.anomaly, feats))
            report["anomaly_score"] = round(norm, 4)
            signals.anomaly = anomaly_to_prob(norm)
        if self.classifier is None and self.anomaly is None:
            report["warnings"].append("nenhum modelo visual treinado ainda (veja README: treino)")

        parsed = ParsedCard()
        if self.ocr is not None:
            with self._ocr_lock:
                lines = self.ocr.read_card(card)
            parsed = parse_card_text(lines, self.s.ocr_lang)
            timer.lap("ocr")
        report["ocr"] = parsed.to_dict()

        ref = None
        if self.reference is not None and (parsed.name or parsed.number):
            ref = self.reference.find_card(parsed.name, parsed.number, parsed.total)  # rede: fora dos locks
            timer.lap("reference_lookup")
        if self.ocr is not None and self.reference is not None:
            from app.pipeline.text_checks import check_text

            result = check_text(parsed, ref)
            signals.text = result.score
            report["text_issues"] = result.issues

        if ref is not None:
            report["reference"] = {
                "id": ref.id, "source": ref.source, "name": ref.name, "set": ref.set_name,
                "number": ref.number, "image": ref.image_url,
            }
            ref_img = self.reference.get_image(ref)  # rede/disco: fora dos locks
            timer.lap("reference_image")
            if ref_img is not None:
                ref_card = cv2.resize(ref_img, (CARD_W, CARD_H), interpolation=cv2.INTER_CUBIC)
                ref_feats = self._embed(ref_card)
                timer.lap("embed")
                a = select_groups(feats, self.embedder.groups, REFERENCE_GROUPS)
                b = select_groups(ref_feats, self.embedder.groups, REFERENCE_GROUPS)
                sims = group_similarity(a, b, len(REFERENCE_GROUPS))
                report["reference"]["similarity"] = round(float(sims.mean()), 4)
                report["reference"]["group_similarity"] = {
                    g: round(float(v), 4) for g, v in zip(REFERENCE_GROUPS, sims)
                }
                if self.ref_anomaly is not None:
                    dists = group_distances(a, b, len(REFERENCE_GROUPS))
                    norm = self.ref_anomaly.normalized(dists)
                    report["reference"]["anomaly_score"] = round(norm, 4)
                    signals.reference = anomaly_to_prob(norm)

        report["preview"] = encode_jpeg_base64(card)
        timer.lap("report")
        return signals, report, card

    # ---------------------------------------------------------------- analyze
    def analyze(self, img_bgr: np.ndarray, *, raw: bytes | None = None, raw_format: str | None = None, sha256: str | None = None) -> dict:
        self.maybe_reload()
        t0 = time.perf_counter()
        signals, report, card = self.compute_signals(img_bgr)
        prob = fuse(signals, self.fusion_model)
        thr = self.thresholds or {"threshold_fake": self.s.threshold_fake, "threshold_real": self.s.threshold_real}
        result = {
            "verdict": verdict(
                prob, signals, thr["threshold_fake"], thr["threshold_real"],
                report["card_detected"], report["texture_reliable"],
            ),
            "fake_probability": None if prob is None else round(prob, 4),
            "signals": {k: (None if v is None else round(v, 4)) for k, v in signals.to_dict().items()},
            "fusion": "treinada" if self.fusion_model is not None else "heurística",
            "models": self.status(),
            **report,
        }
        if self.store is not None:
            try:
                duplicate = self.store.find_duplicate(sha256) if sha256 else None
                result["id"] = duplicate or self.store.save(card, result, raw, raw_format, sha256)
                result["duplicate"] = duplicate is not None
            except OSError:
                log.exception("falha ao persistir inferência")
        result["timings_ms"]["total"] = int((time.perf_counter() - t0) * 1000)
        log.info(
            "analyze id=%s verdict=%s p=%s detected=%s px_mm=%s timings=%s",
            result.get("id"), result["verdict"], result["fake_probability"],
            report["card_detected"], report["px_per_mm"], result["timings_ms"],
        )
        return result
