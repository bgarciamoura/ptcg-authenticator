"""OCR (docTR, PyTorch) em dois passes e extração de campos.

Passe 1: carta inteira (nome, HP, ataques). Passe 2: recorte do rodapé em
resolução cheia — o número da coleção é a menor fonte da carta e se perde
quando o detector reduz a carta inteira para ~1000 px.

O parsing é dirigido por *layouts* declarados (moderno, clássico, promo):
todos são tentados e vence o que extrai mais campos.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

DET_ARCH = "db_mobilenet_v3_large"
RECO_ARCH = "crnn_vgg16_bn"
FOOTER_Y0 = 0.84  # início do recorte do 2º passe


@dataclass
class OcrLine:
    text: str
    conf: float
    # caixa normalizada (0..1) na carta retificada
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass
class ParsedCard:
    name: str | None = None
    hp: int | None = None
    number: str | None = None
    total: str | None = None
    layout: str | None = None
    lines: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def fields_found(self) -> int:
        return sum(v is not None for v in (self.name, self.hp, self.number))


# --------------------------------------------------------------------- idioma
HP_RE = re.compile(r"(?:HP|PS|PV|KP)\s*(\d{2,3})|(\d{2,3})\s*(?:HP|PS|PV|KP)", re.IGNORECASE)
# "Pikachup60", "Charizard 330": HP grudado ao nome quando o OCR junta as caixas
TRAILING_HP_RE = re.compile(r"^(.*?[A-Za-zÀ-ÿ])\s*(?:HP|PS|PV|KP|P|H)?\s*(\d{2,3})$", re.IGNORECASE)

# Rótulos que às vezes vêm grudados ao nome ("BASIC Pikachu", "ESTÁGIO 1 ...").
STOP_TOKENS: dict[str, set[str]] = {
    "en": {"basic", "stage", "evolves", "from", "trainer", "item", "supporter", "stadium", "tool", "energy", "special"},
    "pt": {"básico", "basico", "estágio", "estagio", "evolui", "de", "treinador", "item", "apoiador", "estádio",
           "estadio", "ferramenta", "energia", "especial"},
    "es": {"básico", "basico", "fase", "evoluciona", "de", "entrenador", "objeto", "partidario", "estadio",
           "herramienta", "energía", "energia"},
    "fr": {"base", "niveau", "évolution", "de", "dresseur", "objet", "supporter", "stade", "outil", "énergie"},
    "de": {"basis", "phase", "entwickelt", "sich", "aus", "trainer", "item", "unterstützer", "stadion", "ausrüstung",
           "energie"},
}
EVOLVES_RE = re.compile(r"evolve|evolui|evoluciona|évolution|entwickelt", re.IGNORECASE)


def stop_tokens(lang: str) -> set[str]:
    return STOP_TOKENS["en"] | STOP_TOKENS.get(lang, set()) | {"1", "2"}


# --------------------------------------------------------------------- layouts
_NUM_FRACTION_RE = re.compile(r"\b([A-Z]{0,4}\d{1,3}[a-z]?)\s*/\s*([A-Z]{0,4}\d{1,3})\b")
_NUM_PROMO_RE = re.compile(r"\b((?:SWSH|SVP|SM|XY|BW|HGSS|DP|SV)\s?\d{2,3})\b")


@dataclass(frozen=True)
class Layout:
    name: str
    header: tuple[float, float]  # faixa vertical (y0, y1) com nome e HP
    footer: tuple[float, float]  # faixa vertical onde fica o número
    footer_x_min: float  # 0 = qualquer posição horizontal; 0.5 = só metade direita
    number_re: re.Pattern


LAYOUTS: tuple[Layout, ...] = (
    # Sword & Shield / Scarlet & Violet: número "025/198" embaixo à esquerda
    Layout("modern", (0.0, 0.14), (0.85, 1.0), 0.0, _NUM_FRACTION_RE),
    # WOTC / EX / DP: número embaixo à direita, cabeçalho mais compacto
    Layout("classic", (0.0, 0.12), (0.88, 1.0), 0.5, _NUM_FRACTION_RE),
    # Promos: "SWSH001", "SVP 012" — sem total
    Layout("promo", (0.0, 0.14), (0.85, 1.0), 0.0, _NUM_PROMO_RE),
)


# ------------------------------------------------------------------- engine
def doctr_cache_dir() -> Path:
    return Path(os.environ.get("DOCTR_CACHE_DIR", Path.home() / ".cache" / "doctr")) / "models"


def doctr_weights_present() -> bool:
    cache = doctr_cache_dir()
    return cache.is_dir() and any(cache.glob(f"{DET_ARCH}-*.pt")) and any(cache.glob(f"{RECO_ARCH}-*.pt"))


class CardOCR:
    def __init__(self, allow_download: bool = False):
        if not allow_download and not doctr_weights_present():
            raise RuntimeError(
                "pesos do docTR não estão embutidos na imagem (rebuild com BAKE_MODELS=1) "
                "ou defina ALLOW_MODEL_DOWNLOAD=true"
            )
        from doctr.models import ocr_predictor  # import tardio (pesado)

        self._predictor = ocr_predictor(
            det_arch=DET_ARCH, reco_arch=RECO_ARCH, pretrained=True, assume_straight_pages=True
        )

    def _read(self, img_bgr: np.ndarray) -> list[OcrLine]:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        page = self._predictor([rgb]).export()["pages"][0]
        lines: list[OcrLine] = []
        for block in page["blocks"]:
            for line in block["lines"]:
                words = line["words"]
                if not words:
                    continue
                (x0, y0), (x1, y1) = line["geometry"]
                lines.append(
                    OcrLine(
                        text=" ".join(w["value"] for w in words).strip(),
                        conf=float(np.mean([w["confidence"] for w in words])),
                        x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1),
                    )
                )
        return lines

    def read_card(self, card_bgr: np.ndarray) -> list[OcrLine]:
        """Dois passes: carta inteira + rodapé em resolução cheia."""
        full = self._read(card_bgr)
        h = card_bgr.shape[0]
        footer = self._read(card_bgr[int(FOOTER_Y0 * h) :])
        span = 1.0 - FOOTER_Y0
        remapped = [
            OcrLine(l.text, l.conf, l.x0, FOOTER_Y0 + l.y0 * span, l.x1, FOOTER_Y0 + l.y1 * span) for l in footer
        ]
        # o passe do rodapé substitui o que o passe 1 leu nessa faixa
        kept = [l for l in full if l.cy < FOOTER_Y0]
        return kept + remapped


# --------------------------------------------------------------------- parsing
def _clean_name(text: str, stops: set[str]) -> tuple[str, int | None]:
    """Remove rótulos e HP grudados; devolve (nome, hp_detectado_no_nome)."""
    hp = None
    text = HP_RE.sub(" ", text)
    m = TRAILING_HP_RE.match(text.strip())
    if m and sum(c.isalpha() for c in m.group(1)) >= 3:
        value = int(m.group(2))
        if 30 <= value <= 340 and value % 10 == 0:
            text, hp = m.group(1), value
    tokens = [t for t in re.split(r"\s+", text) if t]
    while tokens and tokens[0].lower().strip(".:") in stops:
        tokens.pop(0)
    return " ".join(tokens).strip(" -:."), hp


def _parse_with_layout(lines: list[OcrLine], layout: Layout, stops: set[str]) -> ParsedCard:
    parsed = ParsedCard(layout=layout.name)
    header = [l for l in lines if layout.header[0] <= l.cy < layout.header[1]]

    for line in header:
        m = HP_RE.search(line.text)
        if m:
            parsed.hp = int(m.group(1) or m.group(2))
            break

    # Nome: maior fonte (altura da caixa) no cabeçalho, majoritariamente letras
    candidates = []
    for line in header:
        if EVOLVES_RE.search(line.text):
            continue
        name, hp_in_name = _clean_name(line.text, stops)
        letters = sum(c.isalpha() for c in name)
        if len(name) >= 3 and letters / max(len(name), 1) > 0.6 and name.lower() not in stops:
            candidates.append((line.height, name, hp_in_name))
    if candidates:
        _, parsed.name, hp_in_name = max(candidates)
        if parsed.hp is None and hp_in_name is not None:
            parsed.hp = hp_in_name

    footer = [
        l for l in lines
        if layout.footer[0] <= l.cy <= layout.footer[1] and l.cx >= layout.footer_x_min
    ]
    for line in sorted(footer, key=lambda l: -l.cy):
        m = layout.number_re.search(line.text)
        if m:
            parsed.number = m.group(1).replace(" ", "")
            parsed.total = m.group(2) if m.re.groups >= 2 else None
            break
    return parsed


def parse_card_text(lines: list[OcrLine], lang: str = "en") -> ParsedCard:
    """Tenta todos os layouts e devolve o que extraiu mais campos."""
    stops = stop_tokens(lang)
    ordered = [l.text for l in sorted(lines, key=lambda l: (l.cy, l.x0))]
    best: ParsedCard | None = None
    for layout in LAYOUTS:
        parsed = _parse_with_layout(lines, layout, stops)
        if best is None or parsed.fields_found() > best.fields_found():
            best = parsed
    assert best is not None
    best.lines = ordered
    return best
