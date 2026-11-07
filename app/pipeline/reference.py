"""Bases de referência: Pokémon TCG API (primária) e TCGdex (fallback, com
PT/ES/FR/DE/JA), com cache em disco de consultas e imagens.

Nenhuma das duas é oficial. A pokemontcg.io costuma atrasar coleções novas;
o TCGdex cobre mais idiomas. Uma carta ausente nas duas NÃO é evidência de
fraude: ``find_card`` devolve None e as checagens tratam como "sem informação".
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np
from rapidfuzz import fuzz

from app.pipeline.imageio import decode_image, sha256_hex

log = logging.getLogger(__name__)

# Só baixa imagens destes hosts (evita SSRF com URLs vindas das APIs)
ALLOWED_IMAGE_HOSTS = {"images.pokemontcg.io", "assets.tcgdex.net"}
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_NOT_FOUND_TTL_S = 24 * 3600  # "não encontrada" expira mais cedo: a base pode indexar depois
_TCGIO_FIELDS = "id,name,hp,number,set,attacks,images,supertype,rarity"


@dataclass
class ReferenceCard:
    id: str
    source: str  # "pokemontcg.io" | "tcgdex"
    name: str
    hp: str | None = None
    number: str | None = None
    printed_total: int | None = None
    set_name: str | None = None
    attacks: list[str] = field(default_factory=list)
    image_url: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ReferenceCard":
        return cls(**d)


def sanitize_name(name: str) -> str:
    """Remove caracteres que poderiam alterar a query Lucene da API."""
    cleaned = re.sub(r"[^\w\s'\-.]", " ", name, flags=re.UNICODE)
    return re.sub(r"\s+", " ", cleaned).strip()[:60]


def normalize_number(number: str | None) -> str | None:
    """'025' -> '25' (as APIs guardam sem zeros à esquerda); 'TG05' fica igual."""
    if not number:
        return None
    if number.isdigit():
        return number.lstrip("0") or "0"
    return number


def _same_number(a: str | None, b: str | None) -> bool:
    return normalize_number(a) is not None and normalize_number(a).lower() == (normalize_number(b) or "").lower()


# ------------------------------------------------------------------- cache
class DiskCache:
    """JSON por chave, com expiração. Sobrevive a restarts (lru_cache não)."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{sha256_hex(key.encode())}.json"

    def get(self, key: str):
        path = self._path(key)
        if not path.exists():
            return None
        try:
            obj = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        if obj.get("expires", 0) < time.time():
            return None
        return obj  # {"value": ..., "expires": ...}

    def set(self, key: str, value, ttl_s: float) -> None:
        try:
            self._path(key).write_text(json.dumps({"value": value, "expires": time.time() + ttl_s}))
        except OSError:
            log.warning("não foi possível gravar cache em %s", self.root)


# ----------------------------------------------------------------- fontes
class PokemonTcgIoSource:
    name = "pokemontcg.io"

    def __init__(self, client: httpx.Client, base_url: str):
        self._client, self._base = client, base_url.rstrip("/")

    def _query(self, q: str) -> list[dict]:
        resp = self._client.get(f"{self._base}/cards", params={"q": q, "pageSize": 50, "select": _TCGIO_FIELDS})
        resp.raise_for_status()
        return resp.json().get("data", [])

    @staticmethod
    def _to_card(c: dict) -> ReferenceCard:
        images = c.get("images") or {}
        return ReferenceCard(
            id=str(c.get("id", "")),
            source="pokemontcg.io",
            name=c.get("name", ""),
            hp=str(c["hp"]) if c.get("hp") else None,
            number=str(c.get("number")) if c.get("number") is not None else None,
            printed_total=(c.get("set") or {}).get("printedTotal"),
            set_name=(c.get("set") or {}).get("name"),
            attacks=[a.get("name", "") for a in c.get("attacks") or []],
            image_url=images.get("large") or images.get("small"),
        )

    def find(self, name: str, number: str | None, total: str | None) -> ReferenceCard | None:
        queries = []
        n = normalize_number(number)
        if n and re.fullmatch(r"[A-Za-z0-9]{1,8}", n):
            queries.append(f'name:"{name}" number:{n}')
        queries.append(f'name:"{name}"')
        for q in queries:
            cards = self._query(q)
            if not cards:
                continue
            if total:
                same_set = [c for c in cards if str((c.get("set") or {}).get("printedTotal")) == total.lstrip("0")]
                if same_set:
                    return self._to_card(same_set[0])
            return self._to_card(cards[0])
        return None


class TcgDexSource:
    name = "tcgdex"

    def __init__(self, client: httpx.Client, base_url: str, lang: str = "en"):
        self._client, self._base, self._lang = client, base_url.rstrip("/"), lang

    def _detail(self, card_id: str) -> ReferenceCard | None:
        if not _SAFE_ID.match(card_id):
            return None
        resp = self._client.get(f"{self._base}/{self._lang}/cards/{card_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        d = resp.json()
        count = (d.get("set") or {}).get("cardCount") or {}
        image = d.get("image")
        return ReferenceCard(
            id=str(d.get("id", card_id)),
            source="tcgdex",
            name=d.get("name", ""),
            hp=str(d["hp"]) if d.get("hp") is not None else None,
            number=str(d.get("localId")) if d.get("localId") is not None else None,
            printed_total=count.get("official"),
            set_name=(d.get("set") or {}).get("name"),
            attacks=[a.get("name", "") for a in d.get("attacks") or []],
            image_url=f"{image}/high.png" if image else None,
        )

    def find(self, name: str, number: str | None, total: str | None) -> ReferenceCard | None:
        resp = self._client.get(f"{self._base}/{self._lang}/cards", params={"name": name})
        resp.raise_for_status()
        briefs = [b for b in resp.json() if fuzz.ratio(b.get("name", "").lower(), name.lower()) >= 90]
        if number:
            briefs = [b for b in briefs if _same_number(number, b.get("localId"))] or briefs
        # o total só está no detalhe: consulta alguns candidatos
        for brief in briefs[:5]:
            card = self._detail(str(brief.get("id", "")))
            if card is None:
                continue
            if total and card.printed_total and str(card.printed_total) != total.lstrip("0"):
                continue
            return card
        return None


# ---------------------------------------------------------------- client
class ReferenceClient:
    def __init__(
        self,
        cache_dir: Path,
        tcg_api_url: str,
        tcg_api_key: str | None = None,
        tcgdex_api_url: str | None = None,
        lang: str = "en",
        timeout: float = 10.0,
        ttl_hours: int = 24 * 7,
        client: httpx.Client | None = None,
    ):
        headers = {"X-Api-Key": tcg_api_key} if tcg_api_key else {}
        self._client = client or httpx.Client(headers=headers, timeout=timeout, follow_redirects=True)
        self._ttl_s = ttl_hours * 3600
        self._queries = DiskCache(cache_dir / "queries")
        self._image_dir = cache_dir / "images"
        self._image_dir.mkdir(parents=True, exist_ok=True)
        self.sources = [PokemonTcgIoSource(self._client, tcg_api_url)]
        if tcgdex_api_url:
            self.sources.append(TcgDexSource(self._client, tcgdex_api_url, lang))

    def find_card(self, name: str | None, number: str | None, total: str | None) -> ReferenceCard | None:
        name = sanitize_name(name or "")
        if not name:
            return None
        key = f"{name}|{normalize_number(number)}|{total}"
        cached = self._queries.get(key)
        if cached is not None:
            return ReferenceCard.from_dict(cached["value"]) if cached["value"] else None

        for source in self.sources:
            try:
                card = source.find(name, number, total)
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                log.warning("fonte %s falhou (%s): %s", source.name, key, exc)
                continue
            if card is not None:
                self._queries.set(key, card.to_dict(), self._ttl_s)
                return card
        self._queries.set(key, None, _NOT_FOUND_TTL_S)
        return None

    def get_image(self, card: ReferenceCard) -> np.ndarray | None:
        if not card.image_url or not _SAFE_ID.match(card.id):
            return None
        parsed = urlparse(card.image_url)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_IMAGE_HOSTS:
            log.warning("host de imagem não permitido: %s", parsed.hostname)
            return None

        cached = self._image_dir / f"{card.source}_{card.id}.png"
        try:
            if cached.exists():
                return decode_image(cached.read_bytes())
            resp = self._client.get(card.image_url)
            resp.raise_for_status()
            if len(resp.content) > _MAX_IMAGE_BYTES:
                return None
            img = decode_image(resp.content)
            cached.write_bytes(resp.content)
            return img
        except (httpx.HTTPError, ValueError, OSError) as exc:
            log.warning("falha ao obter imagem de referência %s: %s", card.id, exc)
            return None
