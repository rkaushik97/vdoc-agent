"""Toy LoRA SFT of Qwen3-VL-4B on ViDoRe V3 Computer Science (English questions).

The data never touches the pharmaceuticals dev/test split. Each example shows 3 pages (the correct
pages, filled up with BM25 near-misses, shuffled) and trains the model to return the reference
answer and cite the correct pages, in the JSON format the dev eval asks for.

    envs/train/bin/python scripts/toy/sft_train.py                 # full toy run
    envs/train/bin/python scripts/toy/sft_train.py --max-steps 2   # smoke test
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import bm25s
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).parent))
from common import MODEL, REPO_ROOT, SYSTEM, TOP_PAGES, question_text, resize  # noqa: E402

from vdoc_agent.runlog import log_run  # noqa: E402

TRAIN_REPO = "vidore/vidore_v3_computer_science"


def build_examples(seed):
    corpus = load_dataset(TRAIN_REPO, "corpus")["test"]
    queries = load_dataset(TRAIN_REPO, "queries")["test"]
    qrels = load_dataset(TRAIN_REPO, "qrels")["test"]

    english = queries.filter(lambda lang: lang == "english", input_columns="language")
    english_ids = set(english["query_id"])
    gold = defaultdict(dict)  # query_id -> {corpus_id: score}
    for qid, cid, score in zip(qrels["query_id"], qrels["corpus_id"], qrels["score"], strict=True):
        if qid in english_ids:
            gold[qid][cid] = score

    page_ids = corpus["corpus_id"]
    row_of = {cid: i for i, cid in enumerate(page_ids)}
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus["markdown"], stopwords="english", show_progress=False))
    positions, _ = retriever.retrieve(
        bm25s.tokenize(english["query"], stopwords="english", show_progress=False), k=20
    )

    rng = random.Random(seed)
    examples = []
    for q, row in zip(english, positions, strict=True):
        correct = sorted(
            gold[q["query_id"]], key=lambda c: -gold[q["query_id"]][c]
        )  # score 2 first
        shown = correct[:2]
        for p in row:  # fill up with BM25 near-misses
            if len(shown) == TOP_PAGES:
                break
            if page_ids[p] not in gold[q["query_id"]]:
                shown.append(page_ids[p])
        rng.shuffle(shown)
        target = {
            "answer": q["answer"],
            "citations": [c for c in shown if c in gold[q["query_id"]]],
            "abstain": False,
        }
        examples.append(
            {
                "images": [resize(corpus[row_of[c]]["image"]) for c in shown],
                "prompt": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": question_text(shown, q["query"])},
                ],
                "completion": [{"role": "assistant", "content": json.dumps(target)}],
            }
        )
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "checkpoints" / "toy-sft-cs"))
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    examples = build_examples(args.seed)
    print(f"{len(examples)} training examples from {TRAIN_REPO}", flush=True)
    print("example target:", examples[0]["completion"][0]["content"][:300], flush=True)

    processor = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa"
    )

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        # language-model layers only: the vision tower names its layers qkv / proj / linear_fc*
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=3,
        bf16=True,
        gradient_checkpointing=True,
        max_length=None,  # never truncate: a cut-off example would lose its answer
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=Dataset.from_list(examples),
        processing_class=processor,
        peft_config=lora,
    )
    trainer.model.print_trainable_parameters()
    result = trainer.train()
    trainer.save_model(args.out)

    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    metrics = {
        "train_loss": result.training_loss,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "steps": trainer.state.global_step,
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "runtime_s": result.metrics.get("train_runtime"),
    }
    Path(args.out, "loss_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    print(json.dumps(metrics, indent=1), flush=True)
    print(
        log_run(
            "toy-sft-cs",
            {**vars(args), "model": MODEL, "train_data": TRAIN_REPO, "lora": lora.to_dict()},
            metrics,
        )
    )


if __name__ == "__main__":
    main()
