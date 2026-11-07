COMPOSE ?= docker compose
GPU := -f compose.yaml -f compose.gpu.yaml
TRAIN := $(COMPOSE) run --rm trainer

.PHONY: build up down logs restart test test-full prepare train fusion pipeline runs review approve-all cleanup shell gpu-up gpu-train

build:      ## Constrói a imagem de inferência (CPU)
	$(COMPOSE) build api

up:         ## Sobe a API em http://localhost:8088 (ou $API_PORT do .env)
	$(COMPOSE) up -d api

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f api

restart:
	$(COMPOSE) restart api

test:       ## Testes unitários (imagem leve)
	$(COMPOSE) run --rm --build tests

test-full:  ## Testes de integração (embedder/OCR reais)
	$(COMPOSE) run --rm tests-full

prepare:    ## Retifica fotos de data/raw (+ feedback aprovado) e extrai embeddings
	$(TRAIN) python -m app.train.prepare

train:      ## Treina os modelos em um novo run e o promove (a API recarrega sozinha)
	$(TRAIN) python -m app.train.train

fusion:     ## Calibra a fusão com data/validation
	$(TRAIN) python -m app.train.fit_fusion

pipeline: prepare train

runs:       ## Lista os runs (* = atual)
	$(TRAIN) python -m app.admin runs

review:     ## Lista rótulos pendentes de revisão
	$(TRAIN) python -m app.admin review

approve-all: ## Aprova todos os rótulos pendentes (confira antes com `make review`)
	$(TRAIN) python -m app.admin review --approve-all

cleanup:    ## Aplica retenção/limite de disco em data/inferences
	$(TRAIN) python -m app.admin cleanup

shell:
	$(TRAIN) bash

gpu-up:
	$(COMPOSE) $(GPU) up -d --build api

gpu-train:
	$(COMPOSE) $(GPU) run --rm trainer python -m app.train.prepare
	$(COMPOSE) $(GPU) run --rm trainer python -m app.train.train
