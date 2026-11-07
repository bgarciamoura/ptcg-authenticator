"""Modelos treináveis, serializados em JSON/NPZ (sem pickle).

Cada modelo guarda ``groups`` (grupos de features que consome) e ``meta``
(identidade do embedder e, na fusão, dos componentes presentes).
``load_if_exists`` recusa modelos cujo meta não bate com o ambiente atual.
A inferência é numpy puro; o scikit-learn só é importado no ``fit``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


class VisualClassifier:
    """PCA → padronização → regressão logística → calibração de Platt.

    Com dezenas de fotos e milhares de dimensões a regressão logística decora
    o treino e devolve 0.01/0.99. O PCA reduz a dimensão para no máximo
    ``n_components`` e a calibração (ajustada em predições out-of-fold)
    transforma os logits em probabilidades honestas. Classe 1 = falsa.
    """

    def __init__(self, groups, meta: dict, C: float = 0.1, n_components: int = 32):
        self.groups = list(groups)
        self.meta = meta
        self.C = C
        self.n_components = n_components
        self.pca_mean: np.ndarray | None = None
        self.components: np.ndarray | None = None  # k x d
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.coef: np.ndarray | None = None
        self.intercept: float = 0.0
        self.calib_a: float = 1.0
        self.calib_b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VisualClassifier":
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X, dtype=np.float64)
        k = max(1, min(self.n_components, X.shape[0] - 1, X.shape[1]))
        pca = PCA(n_components=k, svd_solver="full").fit(X)
        Z = pca.transform(X)
        scaler = StandardScaler().fit(Z)
        lr = LogisticRegression(C=self.C, class_weight="balanced", max_iter=5000)
        lr.fit(scaler.transform(Z), y)

        self.pca_mean, self.components = pca.mean_, pca.components_
        self.mean, self.scale = scaler.mean_, scaler.scale_
        self.coef, self.intercept = lr.coef_[0], float(lr.intercept_[0])
        self.calib_a, self.calib_b = 1.0, 0.0
        return self

    def decision(self, X: np.ndarray) -> np.ndarray:
        """Logits (antes da calibração)."""
        assert self.coef is not None, "modelo não treinado"
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Z = (X - self.pca_mean) @ self.components.T
        return (Z - self.mean) / self.scale @ self.coef + self.intercept

    def calibrate(self, logits: np.ndarray, y: np.ndarray) -> "VisualClassifier":
        """Platt scaling sobre logits OUT-OF-FOLD (nunca os de treino)."""
        from sklearn.linear_model import LogisticRegression

        if len(set(np.asarray(y).tolist())) < 2:
            return self
        lr = LogisticRegression(C=1e4, max_iter=5000).fit(np.asarray(logits).reshape(-1, 1), y)
        self.calib_a, self.calib_b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(self.calib_a * self.decision(X) + self.calib_b)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "type": "visual_classifier",
                    "groups": self.groups,
                    "meta": self.meta,
                    "C": self.C,
                    "n_components": self.n_components,
                    "pca_mean": self.pca_mean.tolist(),
                    "components": self.components.tolist(),
                    "mean": self.mean.tolist(),
                    "scale": self.scale.tolist(),
                    "coef": self.coef.tolist(),
                    "intercept": self.intercept,
                    "calib_a": self.calib_a,
                    "calib_b": self.calib_b,
                }
            )
        )

    @classmethod
    def load(cls, path: Path) -> "VisualClassifier":
        obj = json.loads(path.read_text())
        if obj.get("type") != "visual_classifier":
            raise ValueError("arquivo não é um visual_classifier")
        model = cls(obj["groups"], obj["meta"], obj["C"], obj["n_components"])
        model.pca_mean, model.components = np.array(obj["pca_mean"]), np.array(obj["components"])
        model.mean, model.scale = np.array(obj["mean"]), np.array(obj["scale"])
        model.coef, model.intercept = np.array(obj["coef"]), float(obj["intercept"])
        model.calib_a, model.calib_b = float(obj["calib_a"]), float(obj["calib_b"])
        return model


class KnnAnomaly:
    """Detector de anomalia treinado só com cartas reais (fallback sem referência).

    Score = distância de cosseno média aos k vizinhos reais mais próximos;
    normalizado pelo percentil 95 leave-one-out do banco (>1 = mais estranho
    que 95% das reais). Usa só grupos invariantes + textura.
    """

    def __init__(self, groups, meta: dict, k: int = 5):
        self.groups = list(groups)
        self.meta = meta
        self.k = k
        self.bank: np.ndarray | None = None
        self.threshold: float = 1.0

    def fit(self, X_real: np.ndarray) -> "KnnAnomaly":
        if len(X_real) < 2:
            raise ValueError("são necessárias ao menos 2 cartas reais")
        self.bank = _l2(np.asarray(X_real, dtype=np.float32))
        sims = self.bank @ self.bank.T
        np.fill_diagonal(sims, -np.inf)  # leave-one-out
        k = min(self.k, len(self.bank) - 1)
        top = -np.sort(-sims, axis=1)[:, :k]
        loo_scores = 1.0 - top.mean(axis=1)
        self.threshold = max(float(np.percentile(loo_scores, 95)), 1e-6)
        return self

    def score(self, x: np.ndarray) -> float:
        assert self.bank is not None, "modelo não treinado"
        sims = self.bank @ _l2(np.asarray(x, dtype=np.float32).reshape(-1))
        k = min(self.k, len(self.bank))
        return float(1.0 - np.sort(sims)[-k:].mean())

    def normalized(self, x: np.ndarray) -> float:
        return self.score(x) / self.threshold

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            bank=self.bank,
            threshold=np.float64(self.threshold),
            k=np.int64(self.k),
            groups=np.array(json.dumps(self.groups)),
            meta=np.array(json.dumps(self.meta)),
        )

    @classmethod
    def load(cls, path: Path) -> "KnnAnomaly":
        obj = np.load(path, allow_pickle=False)
        model = cls(json.loads(str(obj["groups"])), json.loads(str(obj["meta"])), int(obj["k"]))
        model.bank, model.threshold = obj["bank"], float(obj["threshold"])
        return model


class ReferenceAnomaly:
    """Anomalia CONDICIONADA à carta oficial.

    Uma foto real difere do scan oficial só pelas condições de captura; uma
    falsa difere também em cor, fonte e enquadramento. Para cada grupo guarda
    o percentil 95 da distância foto-real × oficial vista no treino; o score
    é a média das distâncias normalizadas (>1 = mais longe que 95% das reais).
    Por ser relativo à própria carta, não confunde era/coleção nova com fraude.
    """

    def __init__(self, groups, meta: dict):
        self.groups = list(groups)
        self.meta = meta
        self.thresholds: np.ndarray | None = None
        self.n_fit: int = 0

    def fit(self, dists_real: np.ndarray) -> "ReferenceAnomaly":
        dists_real = np.asarray(dists_real, dtype=np.float64)
        if dists_real.ndim != 2 or len(dists_real) < 5:
            raise ValueError("são necessárias ao menos 5 fotos reais com referência")
        self.thresholds = np.maximum(np.percentile(dists_real, 95, axis=0), 1e-3)
        self.n_fit = int(len(dists_real))
        return self

    def normalized(self, dists: np.ndarray) -> float:
        assert self.thresholds is not None, "modelo não treinado"
        return float(np.mean(np.asarray(dists, dtype=np.float64) / self.thresholds))

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "type": "reference_anomaly",
                    "groups": self.groups,
                    "meta": self.meta,
                    "thresholds": self.thresholds.tolist(),
                    "n_fit": self.n_fit,
                }
            )
        )

    @classmethod
    def load(cls, path: Path) -> "ReferenceAnomaly":
        obj = json.loads(path.read_text())
        if obj.get("type") != "reference_anomaly":
            raise ValueError("arquivo não é um reference_anomaly")
        model = cls(obj["groups"], obj["meta"])
        model.thresholds, model.n_fit = np.array(obj["thresholds"]), int(obj["n_fit"])
        return model


class FusionModel:
    """Regressão logística sobre os sinais (valor + indicador de ausência).

    ``meta`` registra o embedder e quais componentes existiam no ajuste: usar a
    fusão em outra configuração invalida os coeficientes, então o loader recusa.
    """

    def __init__(self, signal_names, meta: dict):
        self.signal_names = tuple(signal_names)
        self.meta = meta
        self.coef: np.ndarray | None = None
        self.intercept: float = 0.0

    def features(self, signals: dict) -> np.ndarray:
        row = []
        for name in self.signal_names:
            v = signals.get(name)
            row += [0.5, 1.0] if v is None else [float(v), 0.0]
        return np.array(row, dtype=np.float64)

    def fit(self, signal_dicts: list[dict], y: np.ndarray) -> "FusionModel":
        from sklearn.linear_model import LogisticRegression

        X = np.stack([self.features(s) for s in signal_dicts])
        lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(X, y)
        self.coef, self.intercept = lr.coef_[0].astype(np.float64), float(lr.intercept_[0])
        return self

    def predict_proba(self, signals: dict) -> float:
        assert self.coef is not None, "modelo não treinado"
        return float(_sigmoid(self.features(signals) @ self.coef + self.intercept))

    def coefficients(self) -> dict[str, dict[str, float]]:
        out = {}
        for i, name in enumerate(self.signal_names):
            out[name] = {"value": round(float(self.coef[2 * i]), 4), "missing": round(float(self.coef[2 * i + 1]), 4)}
        return out

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "type": "fusion",
                    "signal_names": list(self.signal_names),
                    "meta": self.meta,
                    "coef": self.coef.tolist(),
                    "intercept": self.intercept,
                }
            )
        )

    @classmethod
    def load(cls, path: Path) -> "FusionModel":
        obj = json.loads(path.read_text())
        if obj.get("type") != "fusion":
            raise ValueError("arquivo não é um modelo de fusão")
        model = cls(tuple(obj["signal_names"]), obj["meta"])
        model.coef, model.intercept = np.array(obj["coef"]), float(obj["intercept"])
        return model


def load_if_exists(loader, path: Path, expected_meta: dict | None = None):
    """Carrega o modelo se existir e se cada chave de expected_meta bater."""
    if not path.exists():
        return None
    try:
        model = loader(path)
    except Exception:  # arquivo corrompido/versão incompatível
        log.exception("falha ao carregar %s", path)
        return None
    for key, value in (expected_meta or {}).items():
        if model.meta.get(key) != value:
            log.warning("%s ignorado: meta['%s'] difere do ambiente atual", path.name, key)
            return None
    return model
