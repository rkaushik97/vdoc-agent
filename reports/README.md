# Reports

## runs/

One JSON file per run, written by `vdoc_agent.runlog.log_run(name, config, metrics)` as
`runs/<UTC timestamp>_<name>.json`. Each record holds:

- `name`, `timestamp` (UTC, ISO 8601)
- `config`: the full config of the run, as passed in
- `metrics`: the numbers the run produced
- `git.commit`, `git.dirty`: the code that produced it, and whether that code was committed
- `host`, `gpu`, `versions` (`python`, `torch`, `transformers`)

Run files are committed. They are small, and they are the evidence behind every number in the
tables below. A record is never edited by hand; the run is repeated instead.

## Result tables

Markdown tables in this directory, one per question, each built from `runs/` and naming the
run files it was built from. The headline numbers go into the results table in the top-level
README.

| File | Question | Day |
|---|---|---|
| `retrieval.md` | which retriever, fusion and reranker give the best Recall@k and nDCG on dev | 6 |
| `agent.md` | ANLS, relaxed accuracy, citation P/R and abstention per agent configuration | 13 |
| `training.md` | base model vs SFT vs DPO vs GRPO on the same metrics | 21 to 27 |
| `final.md` | the single test-set run for the chosen configuration | 28 |
