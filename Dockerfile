# syntax=docker/dockerfile:1
#
# Estágios:
#   base     python + libs de sistema + dependências leves
#   test     base + pytest + opencv-headless  (testes unitários, sem torch)
#   runtime  base + torch + transformers + docTR, com os pesos dos modelos
#            já baixados na imagem (sem depender de rede em runtime)

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    REFERENCE_CACHE_DIR=/cache/reference

# libgl/glib: exigidas pelo opencv; libgomp: torch
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Usuário sem privilégios (uid 1000 = mesmo dono dos bind mounts no host)
RUN useradd -m -u 1000 app \
    && mkdir -p /cache/huggingface /cache/reference /data /models \
    && chown -R app:app /cache /home/app /data /models

WORKDIR /srv
COPY requirements/base.txt requirements/base.txt
RUN pip install -r requirements/base.txt

# ------------------------------------------------------------------ test
FROM base AS test
COPY requirements/test.txt requirements/test.txt
RUN pip install -r requirements/test.txt
COPY --chown=app:app pyproject.toml .
COPY --chown=app:app app ./app
COPY --chown=app:app tests ./tests
USER app
CMD ["pytest"]

# --------------------------------------------------------------- runtime
FROM base AS runtime
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.4.1
COPY requirements/ml.txt requirements/ml.txt
RUN pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}" \
    && pip install -r requirements/ml.txt

# Pesos embutidos na imagem (DINOv2 via Hugging Face, docTR via CDN da Mindee).
# A API recusa baixar em runtime, salvo ALLOW_MODEL_DOWNLOAD=true.
ARG BAKE_MODELS=1
ARG DINO_MODEL=facebook/dinov2-small
USER app
RUN if [ "$BAKE_MODELS" = "1" ]; then \
      n=0; until python -c "from transformers import AutoModel; AutoModel.from_pretrained('${DINO_MODEL}')"; do \
        n=$((n+1)); [ "$n" -ge 5 ] && exit 1; sleep 10; done; \
      n=0; until python -c "from doctr.models import ocr_predictor; ocr_predictor(det_arch='db_mobilenet_v3_large', reco_arch='crnn_vgg16_bn', pretrained=True)"; do \
        n=$((n+1)); [ "$n" -ge 5 ] && exit 1; sleep 10; done; \
    fi

# só o pytest (sem opencv-headless, que conflitaria com o opencv do docTR)
RUN pip install --user pytest==8.3.3
ENV PATH=/home/app/.local/bin:$PATH

COPY --chown=app:app pyproject.toml .
COPY --chown=app:app app ./app
COPY --chown=app:app tests ./tests

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
