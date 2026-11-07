from app.pipeline.ocr import OcrLine, ParsedCard, parse_card_text
from app.pipeline.reference import ReferenceCard, normalize_number, sanitize_name
from app.pipeline.text_checks import check_text

REF = ReferenceCard(
    id="sv3pt5-25", source="pokemontcg.io", name="Pikachu", hp="60", number="25",
    printed_total=165, set_name="151", attacks=["Thunder Jolt"],
)


def _line(text, y0, y1=None, x0=0.1):
    return OcrLine(text=text, conf=0.99, x0=x0, y0=y0, x1=x0 + 0.25, y1=y1 or y0 + 0.03)


def test_parse_modern_layout():
    lines = [
        _line("BASIC", 0.02, 0.04),
        _line("Pikachu", 0.03, 0.08),
        _line("HP60", 0.03, 0.06, x0=0.7),
        _line("Thunder Jolt 30", 0.65),
        _line("025/165", 0.95),
    ]
    p = parse_card_text(lines)
    assert (p.name, p.hp, p.number, p.total) == ("Pikachu", 60, "025", "165")
    assert p.layout == "modern"


def test_parse_name_with_hp_glued():
    """O OCR às vezes junta nome e HP numa caixa só ('Pikachup60', 'Charizard 330')."""
    p = parse_card_text([_line("Pikachup60", 0.03, 0.08), _line("025/165", 0.95)])
    assert (p.name, p.hp) == ("Pikachu", 60)
    p = parse_card_text([_line("Charizard 330", 0.03, 0.08)])
    assert (p.name, p.hp) == ("Charizard", 330)
    # não confunde um número qualquer com HP
    assert parse_card_text([_line("Porygon2", 0.03, 0.08)]).name == "Porygon2"


def test_parse_promo_layout_without_total():
    lines = [_line("Pikachu", 0.03, 0.08), _line("SWSH 020", 0.95)]
    p = parse_card_text(lines)
    assert (p.number, p.total, p.layout) == ("SWSH020", None, "promo")


def test_parse_strips_stage_prefix_by_language():
    assert parse_card_text([_line("BASIC Pikachu HP 60", 0.03, 0.08)]).name == "Pikachu"
    assert parse_card_text([_line("BÁSICO Pikachu PS 60", 0.03, 0.08)], lang="pt").name == "Pikachu"


def test_check_text_consistent():
    parsed = ParsedCard(name="Pikachu", hp=60, number="025", total="165", lines=["Thunder Jolt 30"])
    result = check_text(parsed, REF)
    assert result.score == 0 and result.issues == []


def test_check_text_flags_inconsistencies():
    parsed = ParsedCard(name="Pikachu", hp=200, number="025", total="198", lines=["Thundr Jlt"])
    result = check_text(parsed, REF)
    assert result.score is not None and result.score >= 0.65
    assert len(result.issues) >= 2


def test_not_found_is_no_information():
    """Ausência nas bases não é evidência de fraude (coleções novas atrasam)."""
    result = check_text(ParsedCard(name="Pikachu", number="001", total="200"), None)
    assert result.score is None and "não encontrada" in result.issues[0]
    assert check_text(ParsedCard(), None).score is None


def test_sanitize_and_normalize():
    assert sanitize_name('Pika"chu) OR name:*') == "Pika chu OR name"
    assert normalize_number("025") == "25"
    assert normalize_number("000") == "0"
    assert normalize_number("TG05") == "TG05"
    assert normalize_number(None) is None
