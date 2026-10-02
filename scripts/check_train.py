"""Smoke test for envs/train: torch, transformers, peft, bitsandbytes, trl.

Every check runs even if an earlier one fails; the exit code is non-zero if any
of them failed.
"""

import gc
import importlib.util
import os
import sys
import tempfile
import time
import traceback

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

MODEL_ID = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
RESULTS = {}


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def lora_config():
    from peft import LoraConfig

    return LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS, task_type="CAUSAL_LM"
    )


def check_torch():
    assert torch.cuda.is_available(), "CUDA is not available to torch"
    major, minor = torch.cuda.get_device_capability(0)
    print(
        f"torch {torch.__version__} | CUDA {torch.version.cuda}"
        f" | cuDNN {torch.backends.cudnn.version()}"
    )
    print(f"device       : {torch.cuda.get_device_name(0)}")
    print(f"capability   : {major}.{minor}")
    print(f"bf16 support : {torch.cuda.is_bf16_supported()}")
    assert torch.cuda.is_bf16_supported(), "bf16 is not supported on this GPU"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(
        f"TF32         : matmul={torch.backends.cuda.matmul.allow_tf32}"
        f" cudnn={torch.backends.cudnn.allow_tf32}"
    )

    n, iters = 8192, 50
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tflops = 2 * n**3 * iters / elapsed / 1e12
    print(f"matmul       : {n}x{n} bf16, {iters} iters in {elapsed:.3f} s -> {tflops:.1f} TFLOPS")
    RESULTS["matmul_tflops"] = tflops


def check_transformers():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda").eval()
    cfg = model.config
    # Newer transformers keep rope_theta inside config.rope_parameters.
    rope_theta = getattr(cfg, "rope_theta", None) or (
        getattr(cfg, "rope_parameters", None) or {}
    ).get("rope_theta")
    print(f"model        : {MODEL_ID} ({next(model.parameters()).dtype})")
    print(f"layers       : {cfg.num_hidden_layers}")
    print(f"heads        : {cfg.num_attention_heads}")
    print(f"KV heads     : {cfg.num_key_value_heads}")
    print(f"hidden size  : {cfg.hidden_size}")
    print(f"vocab        : {cfg.vocab_size}")
    print(f"rope_theta   : {rope_theta}")
    flash = importlib.util.find_spec("flash_attn") is not None
    attention = "flash_attn installed" if flash else "PyTorch SDPA (flash_attn not installed)"
    print(f"attention    : {attention}")

    inputs = tokenizer("The capital of Switzerland is", return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits
    assert torch.isfinite(logits).all(), "non-finite logits"
    next_token = tokenizer.decode(logits[0, -1].argmax())
    print(f"logits shape : {tuple(logits.shape)} (next token: {next_token!r})")
    del model
    free_gpu()


def check_peft():
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda")
    model = get_peft_model(model, lora_config())
    trainable, total = model.get_nb_trainable_parameters()
    print(f"LoRA r=16 on : {', '.join(LORA_TARGETS)}")
    print(f"trainable    : {trainable:,} / {total:,} params ({100 * trainable / total:.3f}%)")
    assert 0 < trainable < total
    del model
    free_gpu()


def check_bitsandbytes():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=quant, device_map={"": 0}
    )
    footprint = model.get_memory_footprint()
    print(f"4-bit NF4    : memory footprint {footprint / 2**20:.1f} MiB")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    inputs = tokenizer("2 + 2 =", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    print(f"4-bit sample : {tokenizer.decode(out[0], skip_special_tokens=True)!r}")
    RESULTS["nf4_mib"] = footprint / 2**20
    del model
    free_gpu()


def check_trl():
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(0)
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"What is {i} + {i + 3}?"},
                {"role": "assistant", "content": f"{i} + {i + 3} = {2 * i + 3}."},
            ]
        }
        for i in range(50)
    ]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda")

    free_gpu()
    torch.cuda.reset_peak_memory_stats()
    with tempfile.TemporaryDirectory() as out_dir:
        args = SFTConfig(
            output_dir=out_dir,
            max_steps=10,
            per_device_train_batch_size=4,
            learning_rate=2e-4,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            max_length=128,
        )
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=Dataset.from_list(rows),
            processing_class=tokenizer,
            peft_config=lora_config(),
        )
        output = trainer.train()

    losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    # lora_B starts at zero, so any non-zero entry proves gradients reached the adapters.
    moved = any(
        p.abs().sum().item() > 0 for name, p in trainer.model.named_parameters() if "lora_B" in name
    )
    print(f"SFT steps    : {trainer.state.global_step}")
    print(
        f"SFT loss     : first {losses[0]:.4f} -> last {losses[-1]:.4f}"
        f" (mean {output.training_loss:.4f})"
    )
    print(f"SFT peak mem : {peak_gb:.2f} GiB (torch.cuda.max_memory_allocated)")
    assert trainer.state.global_step == 10
    assert all(loss == loss and loss != float("inf") for loss in losses), "non-finite loss"
    assert moved, "LoRA weights did not change: no gradient reached the adapters"
    RESULTS["sft_loss"] = losses[-1]
    RESULTS["sft_peak_gib"] = peak_gb
    del trainer, model
    free_gpu()


CHECKS = [
    ("a) torch", check_torch),
    ("b) transformers", check_transformers),
    ("c) peft", check_peft),
    ("d) bitsandbytes", check_bitsandbytes),
    ("e) trl", check_trl),
]


def main() -> int:
    failed = []
    for title, check in CHECKS:
        print(f"\n=== {title} ===")
        try:
            check()
            print(f"--> PASS {title}")
        except Exception:
            traceback.print_exc()
            print(f"--> FAIL {title}")
            failed.append(title)
            free_gpu()

    print("\nRESULT train " + " ".join(f"{key}={value:.3f}" for key, value in RESULTS.items()))
    if failed:
        print(f"[train-check] FAIL: {', '.join(failed)}")
        return 1
    print("[train-check] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
