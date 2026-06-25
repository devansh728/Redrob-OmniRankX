---
title: OmniRank-X
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.21.0
app_file: app.py
pinned: true
license: mit
---

# OmniRank-X — Intelligent Candidate Discovery & Ranking

**Redrob Hackathon Sandbox** · 5-stage CPU-only candidate ranking pipeline.

## What this does

OmniRank-X reads a JSON/JSONL file of candidate profiles, scores each one against a Senior AI Engineer Job Description, and outputs the top-100 ranked candidates — each with a one-to-two sentence reasoning string.

## Pipeline Stages

| Stage | Module | What it does |
|---|---|---|
| 0 | `loader.py` | Streams JSONL / gzip JSONL into memory |
| 1 | `pruner.py` | Drops honeypots, hard disqualifiers, services-only careers |
| 2 | `semantic.py` | ONNX BGE-small embeddings + BM25 + skill alignment → RRF |
| 3 | `trajectory.py` | Career arc scoring (tenure depth, hop penalty, long stints) |
| 4 | `behavioral.py` | Recency, GitHub, notice period, salary fit, open-to-work |
| 5 | `fusion.py` | Weighted sum → sort → top-100 → reasoning strings |

## How to use the sandbox

1. Upload a JSON array of candidate dicts matching the schema, or a JSONL file (one dict per line).
2. Select a config from the dropdown (or upload your own compiled `generated_config.json`).
3. Click **Run Ranking Pipeline**.
4. Download the output CSV (`candidate_id`, `rank`, `score`, `reasoning`).

The sandbox handles up to ~100 candidates. For the full 100k pipeline, run locally via Docker.

## Local one-command run

```bash
# 1. Build
docker build -t omnirank-x .

# 2. Run (mount data and outputs)
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  omnirank-x \
  python main.py data/candidates.jsonl outputs/submission.csv
```

## Tech stack

- **Python 3.11** · **Polars** (streaming JSONL) · **ONNX Runtime** (CPU INT8 inference)
- **sentence-transformers / optimum** (model export, download only) · **rank-bm25**
- **Gradio 4.21** (sandbox UI)

Model weights (`bge-small-en-v1.5` ONNX INT8, ~23 MB) are committed directly to
this repo via **Git LFS** — no cold-start download, model is ready immediately.
As a fallback, if LFS files weren't pulled, the Space auto-downloads from
`optimum/bge-small-en-v1.5` on HuggingFace Hub and caches to `/tmp/`.
