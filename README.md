# Payment Integration Support RAG

This repository contains a local, read-only research pipeline for acquiring,
normalizing, retrieving, and evaluating Stripe documentation. It does not
execute payments, refunds, or Stripe account operations.

The tracked `config/source-manifest.json` contains public Stripe documentation
URLs. `config/retrieval-pilot.yaml` is a development-only retrieval set with no
answer key. Human-reviewed expected facts, prohibited-claim labels, raw page
snapshots, generated indexes, and evaluation outputs remain local and are not
committed.

## Install

Python 3.12 or newer is required. Install the package and all optional runtime
features in an isolated environment:

```bash
python -m pip install -e '.[api,dense,openai]'
```

The commands below use the local `finance` Conda environment used for this
project. After an editable install, the equivalent `payrag ...` console command
can be used without setting `PYTHONPATH`.

## Current workflow

Run commands from the repository root with `PYTHONPATH=src`:

```bash
python3 -m payrag.cli validate-manifest
python3 -m payrag.cli acquire --timeout 10 --attempts 1
python3 -m payrag.cli probe-expansions data/raw/<snapshot-id>
python3 -m payrag.cli normalize data/raw/<snapshot-id> --max-characters 6000
python3 -m payrag.cli validate-normalization \
  data/raw/<snapshot-id> data/normalized/<snapshot-id>
python3 -m payrag.cli build-bm25 \
  data/normalized/<snapshot-id>/chunks.jsonl \
  --output data/indexes/<snapshot-id>/bm25.json \
  --snapshot-id <snapshot-id>
```

Prepare retrieval-only evidence without running a model:

```bash
python3 -m payrag.cli prepare-context \
  data/indexes/<snapshot-id>/bm25.json \
  "Why can a webhook signature fail after JSON reserialization?" \
  --output data/evaluation/<snapshot-id>/context.json \
  --top-k 5 --max-per-source 1
```

Run the development pilot diagnostic:

```bash
python3 -m payrag.cli evaluate-pilot \
  data/indexes/<snapshot-id>/bm25.json \
  --output data/evaluation/<snapshot-id>/bm25-pilot.json \
  --top-k 5
```

The pilot is development data. Its source-level diagnostic is not a formal
benchmark. Snapshot-specific chunk binding, human gold review, model answer
evaluation, and Stripe sandbox testing remain separate work.

## Dense retrieval in the finance environment

The local `finance` conda environment includes Sentence Transformers, Torch,
NumPy, and Transformers. Build and evaluate a local cosine index with:

```bash
conda run -n finance env PYTHONPATH=src python -m payrag.cli normalize \
  data/raw/<snapshot-id> \
  --output data/normalized/<snapshot-id>-c900 \
  --max-characters 900

conda run -n finance env PYTHONPATH=src python -m payrag.cli build-dense \
  data/normalized/<snapshot-id>-c900/chunks.jsonl \
  --output data/indexes/<snapshot-id>/dense-minilm.json \
  --snapshot-id <snapshot-id>

conda run -n finance env PYTHONPATH=src python -m payrag.cli evaluate-pilot-dense \
  data/indexes/<snapshot-id>/dense-minilm.json \
  --output data/evaluation/<snapshot-id>/dense-minilm-pilot.json \
  --top-k 5
```

Dense index construction records the model tokenizer distribution and splits
oversized chunks into traceable windows before embedding. The current MiniLM
index has no silently truncated segments.

## API server

Run the service from the repository root in the `finance` Conda environment:

```bash
conda run -n finance env PYTHONPATH=src python -m payrag.cli serve-api \
  --bm25-index data/indexes/<snapshot-id>/bm25-c900.json \
  --dense-index data/indexes/<snapshot-id>/dense-minilm-c900.json \
  --host 127.0.0.1 \
  --port 8000
```

The service exposes `GET /health`, `POST /v1/context`,
`POST /v1/answers/validate`, and `POST /v1/generate`. BM25 and dense retrieval
work without a generation provider. A missing retriever or generation provider
returns a structured HTTP 503 operational error such as
`operational_error/provider_not_configured`; it is not reported as insufficient
evidence.

To enable generation, provide `OPENAI_API_KEY` through the process environment
and select a model at server startup:

```bash
conda run -n finance env OPENAI_API_KEY="$OPENAI_API_KEY" PYTHONPATH=src \
  python -m payrag.cli serve-api \
  --bm25-index data/indexes/<snapshot-id>/bm25-c900.json \
  --dense-index data/indexes/<snapshot-id>/dense-minilm-c900.json \
  --generation-model YOUR_MODEL_ID
```

Generation uses OpenAI Responses Structured Outputs. Do not send API keys,
client secrets, or payment card numbers in questions. The service masks known
sensitive patterns before retrieval and generation.

## Experiment preparation

Create snapshot-scoped evidence candidates for human review:

```bash
conda run -n finance env PYTHONPATH=src python -m payrag.experiments \
  evidence-candidates \
  --snapshot-id <snapshot-id> \
  --pilot config/retrieval-pilot.yaml \
  --chunks data/normalized/<snapshot-id>-c900/chunks.jsonl \
  --output data/evaluation/<snapshot-id>/evidence-binding-candidates.json
```

The command never edits `retrieval-pilot.yaml` and never promotes candidates to
gold. After candidate generation, prepare the M2 comparison contexts:

```bash
conda run -n finance env PYTHONPATH=src python -m payrag.experiments \
  m2-contexts \
  --snapshot-id <snapshot-id> \
  --pilot config/retrieval-pilot.yaml \
  --source-manifest config/source-manifest.json \
  --bm25-index data/indexes/<snapshot-id>/bm25-c900.json \
  --dense-index data/indexes/<snapshot-id>/dense-minilm-c900.json \
  --evidence-candidates data/evaluation/<snapshot-id>/evidence-binding-candidates.json \
  --output data/evaluation/<snapshot-id>/m2-context-manifest.json
```

The manifest prepares no-context, BM25, and dense inputs. Oracle and evidence
ablation remain blocked until a human accepts the snapshot chunk bindings. The
manifest also remains ineligible for comparison claims until the generation
model, tokenizer, prompt version, and common maximum context token budget are
fixed and actual provider runs are recorded.
