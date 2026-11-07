"""Configuração central, lida de variáveis de ambiente / arquivo .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path("/data")
    models_dir: Path = Path("/models")
    reference_cache_dir: Path = Path("/cache/reference")

    # Ciclo de dados
    store_inferences: bool = True
    inferences_dir: Path = Path("/data/inferences")
    retention_days: int = 180  # inferências sem rótulo mais antigas são apagadas
    max_inferences_gb: float = 5.0  # acima disso apaga as mais antigas sem rótulo
    feedback_token: str | None = None  # se definido, /feedback exige X-Feedback-Token

    # Embedder visual. DINO_MODEL também é build arg: o modelo precisa estar
    # embutido na imagem, salvo ALLOW_MODEL_DOWNLOAD=true.
    device: str = "auto"  # auto | cpu | cuda
    dino_model: str = "facebook/dinov2-small"
    embed_size: int = 224  # múltiplo de 14 (patch do DINOv2)
    allow_model_download: bool = False

    # Resolução mínima da carta na foto para os tiles de textura valerem algo
    # (12 px/mm ≈ carta com 1056 px de altura na foto original).
    min_px_per_mm: float = 12.0

    # OCR (docTR). ocr_lang só afeta os rótulos de layout e o idioma do TCGdex.
    enable_ocr: bool = True
    ocr_lang: str = "en"

    # Bases de referência
    enable_reference: bool = True
    tcg_api_url: str = "https://api.pokemontcg.io/v2"
    tcg_api_key: str | None = None
    enable_tcgdex: bool = True
    tcgdex_api_url: str = "https://api.tcgdex.net/v2"
    reference_timeout: float = 10.0
    reference_cache_ttl_hours: int = 24 * 7

    # API
    max_upload_mb: int = 15
    max_concurrent: int = 2  # análises simultâneas (CPU); o resto espera na fila
    queue_timeout_s: float = 30.0  # tempo máximo na fila antes de 503

    # Limiares padrão; um run treinado pode trazer thresholds.json que os substitui
    threshold_fake: float = 0.65
    threshold_real: float = 0.35


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Instância única, criada só quando pedida (testes passam Settings() direto)."""
    return Settings()
