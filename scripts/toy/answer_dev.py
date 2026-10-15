"""Answer every pharmaceuticals dev question: BM25 top-3 pages -> served model -> JSON answer.

Same pipeline as notebooks/04_answer_v0.ipynb, in the prompt format the toy SFT trains on.
`--model` is the name the vLLM server serves: the base model id, or a LoRA adapter name.

    envs/train/bin/python scripts/toy/answer_dev.py --model Qwen/Qwen3-VL-4B-Instruct \
        --out data/runs/answers-base.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import bm25s
from datasets import load_dataset
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    REPO_ROOT,
    SYSTEM,
    TOP_PAGES,
    answer_schema,
    data_url,
    question_text,
    read_jsonl,
    resize,
    write_jsonl,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--limit", type=int, default=None, help="answer only the first N questions (smoke test)"
    )
    args = ap.parse_args()

    dev = read_jsonl(REPO_ROOT / "evals" / "dev.jsonl")[: args.limit]
    corpus = load_dataset("vidore/vidore_v3_pharmaceuticals", "corpus")["test"]
    page_ids = corpus["corpus_id"]
    row_of = {cid: i for i, cid in enumerate(page_ids)}

    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus["markdown"], stopwords="english", show_progress=False))
    positions, _ = retriever.retrieve(
        bm25s.tokenize([q["query"] for q in dev], stopwords="english", show_progress=False),
        k=TOP_PAGES,
    )
    for q, row in zip(dev, positions, strict=True):
        q["shown"] = [int(page_ids[p]) for p in row]

    client = OpenAI(base_url=args.base_url, api_key="none")

    def answer(q):
        content = [
            {
                "type": "image_url",
                "image_url": {"url": data_url(resize(corpus[row_of[c]]["image"]))},
            }
            for c in q["shown"]
        ]
        content.append({"type": "text", "text": question_text(q["shown"], q["query"])})
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": answer_schema(q["shown"])},
            },
            temperature=0,
            max_tokens=300,
        )
        choice = resp.choices[0]
        try:
            out = json.loads(choice.message.content)
        except json.JSONDecodeError:
            # cut off before the JSON closed, or otherwise broken: a failed answer
            out = {"answer": "", "citations": [], "abstain": False, "raw": choice.message.content}
        out["finish_reason"] = choice.finish_reason
        return out

    with ThreadPoolExecutor(args.workers) as pool:
        preds = list(pool.map(answer, dev))

    rows = []
    for q, p in zip(dev, preds, strict=True):
        correct_pages = {x["corpus_id"] for x in q["pages"]}
        cited = set(p["citations"])
        rows.append(
            {
                "query_id": q["query_id"],
                "query": q["query"],
                "answer": q["answer"],
                "content_type": q["content_type"],
                "query_types": q["query_types"],
                "shown": q["shown"],
                "pred": p,
                "evidence_shown": bool(correct_pages & set(q["shown"])),
                "cite_precision": len(cited & correct_pages) / len(cited) if cited else 0.0,
                "cite_recall": len(cited & correct_pages) / len(correct_pages),
            }
        )
    write_jsonl(args.out, rows)
    print(args.model, "->", args.out, Counter(p["finish_reason"] for p in preds), flush=True)


if __name__ == "__main__":
    main()
