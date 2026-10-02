"""Client half of the vLLM smoke test (the server is started by check_serve.sh).

Sends one streamed chat completion (time to first token, tokens/s) and one
structured-output request constrained by a JSON schema.
"""

import argparse
import json
import sys
import time

from openai import OpenAI

INVOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string"},
        "currency": {"type": "string", "enum": ["USD", "EUR", "CHF"]},
        "total": {"type": "number"},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "amount": {"type": "number"},
                },
                "required": ["description", "amount"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["vendor", "currency", "total", "line_items"],
    "additionalProperties": False,
}

# Qwen3 thinks by default; switch it off so the timings measure plain decoding.
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


def check_chat(client: OpenAI, model: str) -> dict:
    start = time.perf_counter()
    first_token_at = None
    text = []
    usage = None
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Explain what a KV cache does in LLM inference, in about 150 words.",
            }
        ],
        max_tokens=256,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
        extra_body=NO_THINKING,
    )
    for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.model_dump()
        # Reasoning text arrives under a separate key when a reasoning parser is on.
        piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
        if piece:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            text.append(piece)
    end = time.perf_counter()

    assert first_token_at is not None, "stream produced no tokens"
    assert usage is not None, "server did not report usage"
    ttft = first_token_at - start
    decode_time = end - first_token_at
    # The first token is already out when the decode clock starts.
    tok_s = (usage.completion_tokens - 1) / decode_time if decode_time > 0 else float("nan")

    print(f"[chat] completion tokens : {usage.completion_tokens}")
    print(f"[chat] time to first tok : {ttft * 1000:.0f} ms")
    print(f"[chat] decode speed      : {tok_s:.1f} tokens/s")
    print(
        f"[chat] end-to-end        : {usage.completion_tokens / (end - start):.1f} tokens/s"
        f" over {end - start:.2f} s"
    )
    print(f"[chat] text              : {''.join(text)[:160].strip()!r}...")
    return {"ttft_ms": ttft * 1000, "tok_s": tok_s}


def check_structured(client: OpenAI, model: str) -> None:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Extract the invoice as JSON. Invoice from Acme GmbH: "
                "2 widgets at 19.50 EUR each, shipping 5.00 EUR. Total: 44.00 EUR.",
            }
        ],
        max_tokens=256,
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "invoice", "schema": INVOICE_SCHEMA, "strict": True},
        },
        extra_body=NO_THINKING,
    )
    raw = response.choices[0].message.content
    data = json.loads(raw)

    assert set(data) == set(INVOICE_SCHEMA["required"]), f"unexpected keys: {sorted(data)}"
    assert isinstance(data["vendor"], str)
    assert data["currency"] in INVOICE_SCHEMA["properties"]["currency"]["enum"]
    assert isinstance(data["total"], (int, float)) and not isinstance(data["total"], bool)
    assert isinstance(data["line_items"], list)
    for item in data["line_items"]:
        assert set(item) == {"description", "amount"}, f"unexpected item keys: {sorted(item)}"
        assert isinstance(item["description"], str)
        assert isinstance(item["amount"], (int, float)) and not isinstance(item["amount"], bool)

    print(f"[json] schema-valid output: {json.dumps(data)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=120)
    served = [m.id for m in client.models.list().data]
    print(f"[serve] models served: {served}")
    assert args.model in served, f"{args.model} is not being served"

    stats = check_chat(client, args.model)
    check_structured(client, args.model)
    print(f"RESULT serve ttft_ms={stats['ttft_ms']:.0f} tokens_per_s={stats['tok_s']:.1f}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as err:
        print(f"[serve] FAIL: {err}", file=sys.stderr)
        sys.exit(1)
