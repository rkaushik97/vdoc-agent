"""Grade saved answer files with one judge and print them side by side.

Same judge prompt as notebooks 04 and 05. Writes `<file>.judged.jsonl` next to each input and logs
one run per file to reports/runs/.

    envs/train/bin/python scripts/toy/judge.py data/runs/answers-base.jsonl \
        data/runs/answers-toy.jsonl
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from common import JUDGE_MODEL, read_jsonl, write_jsonl  # noqa: E402

from vdoc_agent.runlog import log_run  # noqa: E402

JUDGE = (
    "You grade an answer against a reference answer. "
    "The answer is correct if it states the same key facts as the reference: "
    "numbers, names and dates must match. Extra detail is fine. "
    "Missing or wrong key facts make it incorrect. An empty answer is incorrect."
)
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"correct": {"type": "boolean"}},
    "required": ["correct"],
}


def share(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--judge", default=JUDGE_MODEL)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="none")

    def judge(r):
        prompt = (
            f"Question: {r['query']}\n"
            f"Reference answer: {r['answer']}\n"
            f"Answer to grade: {r['pred']['answer']}"
        )
        resp = client.chat.completions.create(
            model=args.judge,
            messages=[{"role": "system", "content": JUDGE}, {"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "grade", "schema": JUDGE_SCHEMA},
            },
            temperature=0,
            max_tokens=20,
        )
        return json.loads(resp.choices[0].message.content)["correct"]

    columns = ["accuracy", "accuracy_evidence_shown", "accuracy_evidence_not_shown"]
    columns += ["citation_precision", "citation_recall", "abstain_rate", "cut_off_rate"]
    headers = ["acc", "acc|shown", "acc|not", "cite_P", "cite_R", "abstain", "cut off"]
    print(f"{'file':40}" + "".join(f"{h:>10}" for h in headers))
    for path in args.files:
        rows = read_jsonl(path)
        with ThreadPoolExecutor(args.workers) as pool:
            for r, v in zip(rows, pool.map(judge, rows), strict=True):
                r["correct"] = v
        write_jsonl(Path(path).with_suffix(".judged.jsonl"), rows)

        metrics = {
            "accuracy": share(r["correct"] for r in rows),
            "accuracy_evidence_shown": share(r["correct"] for r in rows if r["evidence_shown"]),
            "accuracy_evidence_not_shown": share(
                r["correct"] for r in rows if not r["evidence_shown"]
            ),
            "citation_precision": share(r["cite_precision"] for r in rows),
            "citation_recall": share(r["cite_recall"] for r in rows),
            "abstain_rate": share(r["pred"]["abstain"] for r in rows),
            "cut_off_rate": share(r["pred"].get("finish_reason") == "length" for r in rows),
            "n": len(rows),
        }
        print(f"{Path(path).name:40}" + "".join(f"{metrics[c]:10.3f}" for c in columns), flush=True)
        log_run(
            f"judge-{Path(path).stem}",
            {"answers": str(path), "judge": args.judge, "split": "dev"},
            metrics,
        )


if __name__ == "__main__":
    main()
