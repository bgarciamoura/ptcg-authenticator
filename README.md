# Pokémon Card Authenticator

Triagem automática de cartas Pokémon **reais × falsas** a partir de uma foto,
combinando visão computacional, OCR e comparação com bases de referência.

> É uma ferramenta de **triagem**, não de certificação. Uma foto não captura
> espessura, peso, textura ou o "teste da luz" (camada preta interna do papel).
> E toda a arquitetura é uma hipótese até ser medida com fotos reais — veja
> **Piloto** abaixo.

## Como funciona

```
Foto ─► 1. Detecção + retificação (3 estratégias: cinza, saturação, adaptativo)
        │   → carta 1260×1760 + resolução efetiva (px/mm) da carta na foto
        │   sem carta detectada → INCONCLUSIVO, nenhum sinal calculado
        │   resolução < 12 px/mm → nunca "real" (textura invisível)
        2. Embeddings DINOv2 por grupo
           ├─ regiões (letterbox): full, header, art, text_box, footer, corner
           └─ textura: tiles 224 px em resolução nativa, sem resize
           ├─ Classificador (sem art/full; PCA + LogReg + Platt)   → "visual"
           └─ Anomalia k-NN (sem art/full; só cartas reais)        → "anomaly"
        3. OCR docTR em 2 passes (carta + rodapé em resolução cheia) → nome/HP/número
           └─ pokemontcg.io → TCGdex (fallback, PT/ES/FR/DE) → confere texto → "text"
        4. Imagem oficial × foto: distância por região, calibrada na
           distribuição de fotos REAIS (ReferenceAnomaly)              → "reference"
        5. Fusão (heurística ou treinada) + limiares do run → real / falsa / inconclusiva
        6. Inferência persistida (original + carta + resultado) → /feedback → revisão → treino
```

Todos os sinais têm o mesmo contrato: **probabilidade de ser falsa** em [0, 1]
ou `None` (sem informação).

| Módulo | Arquivo |
|---|---|
| Retificação, resolução, regiões e tiles | `app/pipeline/detect.py` |
| Embeddings DINOv2 e grupos | `app/pipeline/embedder.py` |
| Classificador, anomalias e fusão (JSON/NPZ, sem pickle) | `app/pipeline/models.py` |
| OCR (docTR) e layouts | `app/pipeline/ocr.py` |
| Bases de referência + cache em disco | `app/pipeline/reference.py` |
| Checagens de texto | `app/pipeline/text_checks.py` |
| Fusão e veredito | `app/pipeline/fusion.py` |
| Registro de runs (versionamento) | `app/pipeline/registry.py` |
| Persistência, dedup, rótulos, retenção | `app/pipeline/storage.py` |
| Orquestração, locks, hot reload, timings | `app/pipeline/service.py` |
| API + UI | `app/main.py`, `app/static/index.html` |
| Operação (revisão de rótulos, limpeza, runs) | `app/admin.py` |

## Pré-requisitos

- Docker com Compose v2.24+
- (Opcional) GPU NVIDIA + NVIDIA Container Toolkit
- Internet no **build**: os pesos do DINOv2 e do docTR são baixados e
  embutidos na imagem. Em runtime a API **recusa** baixar modelos
  (`ALLOW_MODEL_DOWNLOAD=false`), então o que roda é o que foi testado.

## Uso rápido

```bash
cp .env.example .env   # opcional
make build
make up                # http://localhost:8088  (API_PORT no .env)
make logs
make test              # unitários, imagem leve (~10 s)
make test-full         # integração: embedder + OCR reais
```

Sem modelos treinados a API já funciona com OCR + bases de referência, mas o
veredito é sempre **inconclusivo**: "real" exige sinal visual treinado e
resolução suficiente.

### Para usuários sem terminal (Windows)

`scripts/windows/` tem launchers de duplo clique. `iniciar.bat` verifica o
WSL, liga o `dockerd` dentro da distro se estiver parado, constrói a imagem
na primeira vez (com progresso visível), sobe o projeto, espera o `/health` e
abre o navegador. `parar.bat`, `status.bat`, `ver-logs.bat` e `atualizar.bat`
completam o conjunto. Os `.bat` chamam PowerShell em `scripts/windows/ps/`;
distro e caminho são deduzidos de onde a pasta está (ou de `ps/config.ps1`
se ela for copiada para a Área de Trabalho). Para Docker Desktop em vez de
Docker nativo no WSL, troque `Start-DockerService` em `ps/common.ps1`.

### API

```bash
curl -F "file=@minha_carta.jpg" http://localhost:8088/analyze
curl -X POST http://localhost:8088/feedback/<id> -H 'content-type: application/json' \
     -d '{"label":"fake","note":"borda amarelada"}'          # fica "pending"
curl http://localhost:8088/health      # 200 ok | 503 loading/error
```

`/analyze` devolve, além do veredito: sinais, `px_per_mm`, `texture_reliable`,
`quad` detectado, OCR, referência encontrada, `timings_ms` por etapa,
`duplicate` (upload já visto) e `request_id` (também no header `X-Request-ID`).
Documentação interativa em `/docs`. No máximo `MAX_CONCURRENT` análises rodam
ao mesmo tempo; o excedente espera `QUEUE_TIMEOUT_S` e recebe 503.

## Piloto (faça isto antes de qualquer outra coisa)

1. Fotografe **20 cartas reais e 10 falsas, pareadas** (cada falsa tem a
   mesma carta real), 2–3 fotos cada, de perto (carta ≥ 1100 px de altura na
   foto), fundo liso.
2. `make prepare` e **abra `data/processed/`**: se a retificação errou,
   nada do resto vale.
3. `make train` e leia `models/runs/<run>/metrics.json`: `oof_at_0.5.pr_auc`
   é o número que diz se a premissa (DINOv2 enxerga a diferença) se sustenta.
   `manifest.json` traz `pairing_ratio`; abaixo de 0,5 o classificador pode
   estar aprendendo *qual* carta é.

## Treino

### 1. Organize as fotos (uma pasta por carta física, pareadas)

```
data/raw/real/pikachu-151-025/IMG_001.jpg
data/raw/real/pikachu-151-025/IMG_002.jpg
data/raw/fake/pikachu-151-025/IMG_010.jpg     ← mesmo nome da pasta real
```

A validação cruzada separa por carta física. Inferências com rótulo
**aprovado** (`make review` → `make approve-all`) entram automaticamente.

### 2. Rode o pipeline

```bash
make prepare   # retifica, extrai embeddings e distâncias à carta oficial (OCR + rede)
make train     # novo run em models/runs/<timestamp>/ e promove; a API recarrega sozinha
make runs      # lista runs (* = atual); para voltar atrás, edite models/latest.json
```

`prepare --no-reference` pula OCR/rede (mais rápido, sem `ReferenceAnomaly`).

O que `train` produz, conforme os dados:

| Modelo | Precisa de | Grupos |
|---|---|---|
| `anomaly.npz` (k-NN) | ≥ 5 fotos reais | invariantes + textura |
| `classifier.json` | reais e falsas, ≥ 2 cartas físicas por classe | invariantes + textura |
| `ref_anomaly.json` | ≥ 5 fotos reais com carta oficial encontrada | regiões semânticas |
| `thresholds.json` | ≥ 40 fotos, ≥ 10 por classe | limiares da curva PR out-of-fold |

Avalie por `recall_fake` e `pr_auc` em `metrics.json`, não pela acurácia.
O classificador usa PCA (≤ 32 componentes) e calibração de Platt sobre
predições out-of-fold, para não devolver 0,01/0,99 com 30 fotos.

### 3. (Opcional) Calibre a fusão

Fotos de **outras** cartas em `data/validation/{real,fake}/<carta>/`:

```bash
make fusion    # grava fusion.json no run atual; a API recarrega sozinha
```

A fusão registra os componentes presentes e é **ignorada** se a configuração
mudar (ex.: OCR desligado), em vez de dar respostas erradas.

## Ciclo de dados

- Cada análise grava `data/inferences/<id>/` com `original.*`, `card.jpg` e
  `result.json`. Uploads idênticos (SHA-256) reaproveitam o id.
- `/feedback` grava o rótulo como **pending**. Só rótulos aprovados por um
  operador (`make review`, `make approve-all`) entram no treino — isso impede
  envenenamento do dataset por quem acessa a API. `FEEDBACK_TOKEN` adiciona
  um token obrigatório no header `X-Feedback-Token`.
- Retenção: inferências **sem rótulo** mais antigas que `RETENTION_DAYS` ou
  acima de `MAX_INFERENCES_GB` são apagadas no start da API e em `make cleanup`.
  Rotuladas nunca são apagadas automaticamente.

## GPU

```bash
make gpu-up      # API com torch CUDA
make gpu-train   # prepare + train na GPU
```

Só o DINOv2 e o docTR usam a GPU.

## Configuração (`.env`)

| Variável | Padrão | Descrição |
|---|---|---|
| `API_PORT` | `8088` | Porta no host (só 127.0.0.1) |
| `DINO_MODEL` | `facebook/dinov2-small` | Também é build arg: embutido na imagem. Trocar exige `make build` e re-treino |
| `ALLOW_MODEL_DOWNLOAD` | `false` | Permite baixar pesos em runtime (não recomendado) |
| `MIN_PX_PER_MM` | `12` | Resolução mínima da carta na foto para o veredito "real" |
| `OCR_LANG` | `en` | Idioma dos rótulos de layout e do TCGdex (`pt`, `es`, `fr`, `de`) |
| `ENABLE_TCGDEX` | `true` | Fallback de referência |
| `MAX_CONCURRENT` / `QUEUE_TIMEOUT_S` | `2` / `30` | Backpressure |
| `STORE_INFERENCES` / `RETENTION_DAYS` / `MAX_INFERENCES_GB` | `true` / `180` / `5` | Ciclo de dados |
| `FEEDBACK_TOKEN` | vazio | Protege `/feedback` |
| `THRESHOLD_FAKE` / `THRESHOLD_REAL` | `0.65` / `0.35` | Padrões; `thresholds.json` do run os substitui |

## Segurança e privacidade

- API só em `127.0.0.1`; para expor, proxy reverso com autenticação e HTTPS.
- Uploads: tipo e tamanho validados, verificação do Pillow, limite de 25 MP
  (decompression bomb), decodificação só após obter um slot de análise.
- Imagens de referência só de `images.pokemontcg.io` e `assets.tcgdex.net`
  (anti-SSRF); queries sanitizadas; cache em disco com expiração.
- Container sem privilégios; `data/` e `models/` somente leitura na API,
  exceto `data/inferences/`.
- Modelos em JSON/NPZ, `allow_pickle=False`.
- **As fotos enviadas ficam guardadas** para rotulagem. `STORE_INFERENCES=false`
  desliga; `RETENTION_DAYS` limita.

## Solução de problemas

- **Build falha baixando pesos** — o build tenta 5 vezes; rode `make build` de
  novo (camadas prontas ficam em cache).
- **`modelo não está embutido na imagem`** — `.env` pede um `DINO_MODEL`
  diferente do que foi embutido. Rode `make build` (o compose passa o mesmo
  valor como build arg) ou volte o `.env`.
- **`port is already allocated`** — mude `API_PORT` no `.env`.
- **`/health` 503 `loading`** — normal nos primeiros ~20 s; `error` indica
  falha ao carregar modelos (`make logs`).
- **Veredito sempre inconclusivo** — sem modelos treinados, ou fotos com
  `px_per_mm` baixo. O resultado diz qual.

## Limitações conhecidas

- Nenhum modelo foi avaliado com fotos reais ainda: a premissa de que o DINOv2
  (treinado para semântica) percebe textura de impressão **não foi validada**.
- A anomalia k-NN ainda confunde "era desconhecida" com "falsa" quando o banco
  não tem cartas daquela era; `ReferenceAnomaly`, que compara com a própria
  carta oficial, é a alternativa robusta e exige referência encontrada.
- OCR/parsing cobrem layouts moderno, clássico e promo; japonês não é lido.
- Embedder e OCR rodam no mesmo processo da API (locks separados, rede fora
  dos locks, backpressure). Para carga alta, separe em workers.
- Verso da carta ainda não é analisado.
