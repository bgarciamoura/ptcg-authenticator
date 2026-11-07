import io
import json

import httpx
import numpy as np
from PIL import Image

from app.pipeline.reference import ReferenceCard, ReferenceClient

TCGIO = "https://api.pokemontcg.io/v2"
TCGDEX = "https://api.tcgdex.net/v2"


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (63, 88), (10, 200, 10)).save(buf, format="PNG")
    return buf.getvalue()


class Backend:
    """Simula as duas APIs; registra as chamadas."""

    def __init__(self, tcgio_cards=(), tcgdex_briefs=(), tcgdex_details=None, fail_tcgio=False):
        self.tcgio_cards = list(tcgio_cards)
        self.tcgdex_briefs = list(tcgdex_briefs)
        self.tcgdex_details = tcgdex_details or {}
        self.fail_tcgio = fail_tcgio
        self.calls: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.calls.append(url)
        if url.startswith(TCGIO):
            if self.fail_tcgio:
                return httpx.Response(500)
            return httpx.Response(200, json={"data": self.tcgio_cards})
        if url.startswith(f"{TCGDEX}/en/cards/"):
            card_id = url.rsplit("/", 1)[1]
            detail = self.tcgdex_details.get(card_id)
            return httpx.Response(200, json=detail) if detail else httpx.Response(404)
        if url.startswith(f"{TCGDEX}/en/cards"):
            return httpx.Response(200, json=self.tcgdex_briefs)
        if "images.pokemontcg.io" in url or "assets.tcgdex.net" in url:
            return httpx.Response(200, content=_png_bytes(), headers={"content-type": "image/png"})
        return httpx.Response(404)


def _client(tmp_path, backend: Backend, tcgdex=True) -> ReferenceClient:
    http = httpx.Client(transport=httpx.MockTransport(backend.handler))
    return ReferenceClient(tmp_path, TCGIO, None, TCGDEX if tcgdex else None, "en", client=http)


TCGIO_CARD = {
    "id": "sv3pt5-25", "name": "Pikachu", "hp": "60", "number": "25",
    "set": {"printedTotal": 165, "name": "151"}, "attacks": [{"name": "Charge"}],
    "images": {"small": "https://images.pokemontcg.io/sv3pt5/25.png", "large": "https://images.pokemontcg.io/sv3pt5/25_hires.png"},
}
TCGDEX_DETAIL = {
    "id": "sv03.5-025", "localId": "025", "name": "Pikachu", "hp": 60, "image": "https://assets.tcgdex.net/en/sv/sv03.5/025",
    "set": {"id": "sv03.5", "name": "151", "cardCount": {"official": 165, "total": 207}},
    "attacks": [{"name": "Charge"}, {"name": "Pika Punch"}],
}


def test_primary_source_and_disk_cache(tmp_path):
    backend = Backend(tcgio_cards=[TCGIO_CARD])
    client = _client(tmp_path, backend)
    card = client.find_card("Pikachu", "025", "165")
    assert isinstance(card, ReferenceCard)
    assert (card.source, card.number, card.printed_total, card.attacks) == ("pokemontcg.io", "25", 165, ["Charge"])
    n = len(backend.calls)
    # segunda consulta vem do cache em disco, inclusive em outra instância
    assert _client(tmp_path, backend).find_card("Pikachu", "025", "165").id == card.id
    assert len(backend.calls) == n


def test_fallback_to_tcgdex_when_primary_fails_or_is_empty(tmp_path):
    briefs = [{"id": "sv03.5-025", "localId": "025", "name": "Pikachu", "image": "https://assets.tcgdex.net/en/sv/sv03.5/025"},
              {"id": "swsh4-43", "localId": "43", "name": "Pikachu", "image": "https://assets.tcgdex.net/en/swsh/swsh4/43"}]
    backend = Backend(fail_tcgio=True, tcgdex_briefs=briefs, tcgdex_details={"sv03.5-025": TCGDEX_DETAIL})
    card = _client(tmp_path, backend).find_card("Pikachu", "25", "165")
    assert card is not None and card.source == "tcgdex"
    assert (card.number, card.printed_total, card.hp, card.image_url) == ("025", 165, "60", "https://assets.tcgdex.net/en/sv/sv03.5/025/high.png")
    assert card.attacks == ["Charge", "Pika Punch"]


def test_not_found_is_cached_and_returns_none(tmp_path):
    backend = Backend()
    client = _client(tmp_path, backend)
    assert client.find_card("Pikachu", "999", None) is None
    n = len(backend.calls)
    assert client.find_card("Pikachu", "999", None) is None
    assert len(backend.calls) == n
    assert client.find_card("", None, None) is None  # sem nome, sem consulta


def test_image_fetch_allowlist_and_cache(tmp_path):
    backend = Backend()
    client = _client(tmp_path, backend)
    card = ReferenceCard(id="x-1", source="tcgdex", name="P", image_url="https://assets.tcgdex.net/en/x/1/high.png")
    img = client.get_image(card)
    assert isinstance(img, np.ndarray) and img.shape == (88, 63, 3)
    assert (tmp_path / "images" / "tcgdex_x-1.png").exists()
    n = len(backend.calls)
    assert client.get_image(card) is not None and len(backend.calls) == n  # cache

    evil = ReferenceCard(id="x-2", source="tcgdex", name="P", image_url="https://evil.example.com/a.png")
    assert client.get_image(evil) is None
    plain = ReferenceCard(id="x-3", source="tcgdex", name="P", image_url="http://assets.tcgdex.net/a.png")
    assert client.get_image(plain) is None  # só https
    assert client.get_image(ReferenceCard(id="../x", source="tcgdex", name="P", image_url="https://assets.tcgdex.net/a.png")) is None


def test_reference_card_roundtrip():
    card = ReferenceCard(id="a", source="s", name="N", attacks=["x"])
    assert ReferenceCard.from_dict(json.loads(json.dumps(card.to_dict()))) == card
