# Evals

Question sets over the corpus, with page-level citations, scored by `vdoc_agent.eval`.
Nothing is here yet: the questions are written once the corpus lands (day 5) and the split is
made right then, once.

## Files

| File | What |
|---|---|
| `dev.jsonl` | 30% of the questions. Used for every iteration: prompts, retrieval settings, agent budgets, training-data choices. |
| `test.jsonl` | 70% of the questions. Frozen: not read, not inspected, not scored until project day 28. |
| `split.json` | How the split was made: seed, stratification key, and the SHA-256 of `test.jsonl` at freeze time. |

One JSON object per line. Fields that are already settled: `question`, `answers` (all accepted
forms), `doc_id`, `pages` (where the evidence is), `answer_type` (span, number, yes/no, or
unanswerable). Unanswerable questions are in both halves so abstention can be scored.

## Split rules

- One split, made once, with a fixed seed, stratified by source document: no document appears
  in both halves. 30% dev, 70% test.
- Dev is the only set touched before day 28. Every number in `reports/` before then is a dev
  number and is labelled as one.
- Test is scored once, on day 28, with the configuration chosen on dev. Before scoring, the
  SHA-256 in `split.json` is checked against `test.jsonl`; a mismatch voids the run.
- If the test file has to change before day 28 (a broken record, a corrupt file), the change,
  the date and the new hash are recorded here, and the old hash stays in the history.
