"""API HTTP (FastAPI) + interface web simples."""
from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.pipeline.imageio import decode_image, image_format, sha256_hex
from app.pipeline.service import Analyzer
from app.pipeline.storage import ID_RE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
STATIC_DIR = Path(__file__).parent / "static"

settings = get_settings()
analyzer = Analyzer(settings)
# Backpressure: só max_concurrent análises ao mesmo tempo; o resto espera até
# queue_timeout_s e recebe 503. Evita 40 imagens decodificadas na memória.
_slots = asyncio.Semaphore(settings.max_concurrent)


def _load_models() -> None:
    try:
        analyzer.load()
    except Exception as exc:  # noqa: BLE001 - qualquer falha deve aparecer no /health
        log.exception("falha ao carregar modelos")
        analyzer.load_error = str(exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Carrega em thread para o servidor responder /health ("loading") enquanto
    # carrega os modelos; /analyze devolve 503 até terminar.
    threading.Thread(target=_load_models, name="model-loader", daemon=True).start()
    yield


app = FastAPI(title="Pokémon Card Authenticator", version="0.3.0", lifespan=lifespan)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


class Feedback(BaseModel):
    label: Literal["real", "fake"]
    note: str | None = Field(default=None, max_length=500)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> JSONResponse:
    if analyzer.load_error:
        return JSONResponse({"status": "error", "detail": analyzer.load_error}, status_code=503)
    if not analyzer.loaded:
        return JSONResponse({"status": "loading"}, status_code=503)
    return JSONResponse({"status": "ok", "models": analyzer.status()})


@app.post("/analyze")
async def analyze(request: Request, file: UploadFile = File(...)) -> dict:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "envie JPEG, PNG ou WEBP")
    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"arquivo maior que {settings.max_upload_mb} MB")
    try:
        fmt = image_format(data)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not analyzer.loaded:
        raise HTTPException(503, "modelos ainda carregando, tente em instantes")

    try:
        await asyncio.wait_for(_slots.acquire(), timeout=settings.queue_timeout_s)
    except TimeoutError:
        raise HTTPException(503, "servidor ocupado, tente novamente em instantes") from None
    try:
        # decodifica só depois de ter o slot: limita a memória em uso
        img = await run_in_threadpool(decode_image, data)
        result = await run_in_threadpool(
            analyzer.analyze, img, raw=data, raw_format=fmt, sha256=sha256_hex(data)
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        _slots.release()
    result["request_id"] = request.state.request_id
    return result


@app.post("/feedback/{inference_id}")
def feedback(
    inference_id: str,
    body: Feedback,
    request: Request,
    x_feedback_token: str | None = Header(default=None),
) -> dict:
    """Registra o rótulo humano de uma inferência (fica 'pending' até revisão)."""
    if analyzer.store is None:
        raise HTTPException(503, "armazenamento de inferências desativado")
    if settings.feedback_token and x_feedback_token != settings.feedback_token:
        raise HTTPException(401, "token de feedback inválido")
    if not ID_RE.match(inference_id):
        raise HTTPException(400, "id inválido")
    source = request.client.host if request.client else None
    try:
        saved = analyzer.store.set_label(inference_id, body.label, body.note, source)
    except KeyError:
        raise HTTPException(404, "inferência não encontrada") from None
    return {"id": inference_id, **saved}
