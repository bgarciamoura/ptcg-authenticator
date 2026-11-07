"""Confronta o texto lido por OCR com os dados oficiais da carta."""
from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz import fuzz

from app.pipeline.ocr import ParsedCard
from app.pipeline.reference import ReferenceCard, normalize_number

# Peso de cada inconsistência na suspeita (0..1). Valores iniciais heurísticos.
W_NAME = 0.4
W_HP = 0.5
W_NUMBER = 0.3
W_TOTAL = 0.5
W_ATTACK = 0.25


@dataclass
class TextCheckResult:
    score: float | None  # suspeita 0..1 (None = sem informação)
    issues: list[str] = field(default_factory=list)


def check_text(parsed: ParsedCard, ref: ReferenceCard | None) -> TextCheckResult:
    if not parsed.name and not parsed.number:
        return TextCheckResult(score=None, issues=["OCR não conseguiu ler nome nem número"])
    if ref is None:
        # Ausência na base NÃO é evidência: coleções recentes e cartas de outros
        # idiomas demoram a ser indexadas. Fica registrado, sem pontuar.
        where = f"{parsed.number}/{parsed.total}" if parsed.total else (parsed.number or "?")
        return TextCheckResult(
            score=None,
            issues=[f"carta '{parsed.name}' ({where}) não encontrada nas bases de referência (pode ser coleção recente)"],
        )

    issues: list[str] = []
    score = 0.0

    if parsed.name and fuzz.ratio(parsed.name.lower(), ref.name.lower()) < 85:
        issues.append(f"nome lido '{parsed.name}' difere do oficial '{ref.name}'")
        score += W_NAME

    if parsed.hp is not None and ref.hp and str(parsed.hp) != str(ref.hp):
        issues.append(f"HP lido {parsed.hp} ≠ oficial {ref.hp}")
        score += W_HP

    if parsed.number and ref.number and normalize_number(parsed.number) != normalize_number(ref.number):
        issues.append(f"número lido {parsed.number} ≠ oficial {ref.number}")
        score += W_NUMBER

    if parsed.total and ref.printed_total and parsed.total.lstrip("0") != str(ref.printed_total):
        issues.append(f"total da coleção lido {parsed.total} ≠ oficial {ref.printed_total}")
        score += W_TOTAL

    # Ataques: cada nome oficial deve aparecer (aproximadamente) em alguma linha
    full_text = [l.lower() for l in parsed.lines]
    for attack in ref.attacks:
        name = attack.lower()
        if name and full_text and max(fuzz.partial_ratio(name, t) for t in full_text) < 80:
            issues.append(f"ataque '{attack}' não encontrado/escrito diferente")
            score += W_ATTACK

    return TextCheckResult(score=min(score, 1.0), issues=issues)
