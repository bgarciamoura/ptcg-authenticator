"""Combina os sinais em uma probabilidade final de a carta ser falsa.

Contrato: TODOS os sinais são "probabilidade de ser falsa" em [0, 1] ou None.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

SIGNAL_NAMES = ("visual", "anomaly", "text", "reference")

# Pesos usados enquanto não existe fusão treinada (python -m app.train.fit_fusion).
# ``reference`` só existe quando há um ReferenceAnomaly treinado em fotos reais,
# ou seja, já vem calibrado pela distribuição real foto × scan oficial.
HEURISTIC_WEIGHTS = {"visual": 0.45, "anomaly": 0.20, "text": 0.15, "reference": 0.20}


@dataclass
class Signals:
    visual: float | None = None  # classificador supervisionado
    anomaly: float | None = None  # k-NN contra o banco de reais
    text: float | None = None  # OCR × base oficial
    reference: float | None = None  # distância à imagem oficial, calibrada

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


def anomaly_to_prob(normalized_score: float) -> float:
    """score normalizado 1.0 (percentil 95 das reais) vira 0.5."""
    return 1.0 / (1.0 + math.exp(-4.0 * (normalized_score - 1.0)))


def fuse(signals: Signals, fusion_model=None) -> float | None:
    if fusion_model is not None:
        return fusion_model.predict_proba(signals.to_dict())
    available = {n: getattr(signals, n) for n in HEURISTIC_WEIGHTS if getattr(signals, n) is not None}
    if not available:
        return None
    total = sum(HEURISTIC_WEIGHTS[n] for n in available)
    return sum(HEURISTIC_WEIGHTS[n] * v for n, v in available.items()) / total


def verdict(
    prob_fake: float | None,
    signals: Signals,
    threshold_fake: float,
    threshold_real: float,
    card_detected: bool = True,
    texture_reliable: bool = True,
) -> str:
    if not card_detected or prob_fake is None:
        return "inconclusiva"
    if prob_fake >= threshold_fake:
        return "falsa"
    # "real" exige evidência visual treinada E resolução suficiente: uma foto
    # pequena esconde justamente os defeitos de impressão que entregam a falsa.
    has_visual = any(getattr(signals, n) is not None for n in ("visual", "anomaly", "reference"))
    if prob_fake <= threshold_real and has_visual and texture_reliable:
        return "real"
    return "inconclusiva"
