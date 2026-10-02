# vdoc-agent

Agentic RAG over visually rich PDFs: hybrid retrieval, page-cited answers, a validated eval
harness, and Qwen3-VL-4B fine-tuned with LoRA, DPO and GRPO.

## Goal

Answer questions about a corpus of visually rich PDFs (tables, charts, forms, scanned pages)
with answers that cite the page they come from, abstain when the corpus does not contain the
answer, and are scored by an eval harness that is itself checked against human judgement. Then
measure how much of that a 4B vision-language model learns through SFT, DPO and GRPO on LoRA
adapters written from scratch.

## Planned architecture

- **Ingest** (`vdoc_agent.ingest`): PDF -> page PNG + page text (PyMuPDF, Docling) -> one corpus parquet.
- **Retrieval** (`vdoc_agent.retrieval`): BM25, dense (Qwen3-Embedding + FAISS), ColQwen2.5 MaxSim over page images, RRF fusion, rerankers; Recall@k, MRR, nDCG.
- **Agent** (`vdoc_agent.agent`): tool schemas, agent loop, harness (budgets, retries, replay cache, tracing), LangGraph graph, MCP server. The model is served by vLLM from `envs/serve` and called over HTTP; no other code imports vLLM.
- **Eval** (`vdoc_agent.eval`): ANLS, relaxed accuracy, citation precision/recall, abstention, LLM judge validated with Cohen's kappa, bootstrap confidence intervals. Split rules in [evals/README.md](evals/README.md).
- **Train** (`vdoc_agent.train`): LoRA from scratch, SFT data builder, SFT / DPO / GRPO scripts, run from `envs/train`.
- **Run log** (`vdoc_agent.runlog`): every run writes config, metrics and provenance to `reports/runs/` ([reports/README.md](reports/README.md)).

## Results

Dev-set numbers until day 28; the test set is scored once, then.

| System | Recall@5 | ANLS | Relaxed acc. | Citation P / R | Abstention F1 |
|---|---|---|---|---|---|
| | | | | | |

## Setup

Environments, cluster notes and smoke checks: [SETUP.md](SETUP.md). Day to day:

```
make test        # pytest, from envs/train
make lint        # ruff check + ruff format --check
make versions    # driver, CUDA and package versions per env
make help        # everything else, including the targets that are not there yet
```

## Layout

```
src/vdoc_agent/   ingest, retrieval, agent, eval, train, runlog.py
configs/          example.yaml
evals/            dev / test question sets (see evals/README.md)
reports/          runs/ (one JSON per run) and result tables
notebooks/        exploration only; nothing imports from here
tests/            pytest
data/             corpus, indexes, caches (git-ignored)
checkpoints/      adapters and trainer state (git-ignored)
envs/             uv environments serve and train (git-ignored, lock files tracked)
scripts/          environment setup and smoke checks
```
