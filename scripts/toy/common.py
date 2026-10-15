"""Settings and prompt format shared by the toy SFT run and its evals.

Training and evaluation must build the prompt the same way, so both import it from here:
page images first, then one text block naming the pages in order, then the question.
"""

import base64
import io
import json
from pathlib import Path

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
JUDGE_MODEL = "Qwen/Qwen3-VL-8B-Instruct-FP8"
MAX_SIDE = 1280  # longest side of each page image, in pixels
TOP_PAGES = 3  # pages shown per question

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS = REPO_ROOT / "data" / "runs"

SYSTEM = (
    "You answer questions about documents using only the pages you are given. "
    "Answer in one or two sentences. "
    "In citations, list the ids of the pages that contain the answer. "
    "If the pages do not contain the answer, set abstain to true and leave answer empty. "
    'Reply with JSON only: {"answer": string, "citations": [page ids], "abstain": boolean}.'
)


def resize(img):
    """RGB copy of a page image with its longest side at most MAX_SIDE."""
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def question_text(page_ids, question):
    """The text that follows the page images."""
    pages = ", ".join(f"Page {cid}" for cid in page_ids)
    return f"The pages above are, in order: {pages}.\nQuestion: {question}"


def data_url(img):
    """A PIL image as a base64 JPEG data URL, for the OpenAI-style API."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def answer_schema(page_ids):
    """JSON schema for the answer; citations may only name the pages shown."""
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "integer", "enum": list(page_ids)}},
            "abstain": {"type": "boolean"},
        },
        "required": ["answer", "citations", "abstain"],
    }


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))
