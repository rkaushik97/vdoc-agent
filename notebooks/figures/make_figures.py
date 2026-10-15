"""Figures for notebooks/03_search_v0.ipynb: the pipeline and the four search metrics.

    envs/train/bin/python notebooks/figures/make_figures.py

One worked example runs through the metric figures: a search returns 10 pages, and the question's
3 relevant pages sit at ranks 2 (grade 2), 4 (grade 1) and 7 (grade 1).
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent

# colours: one blue for "relevant" (dark = grade 2, light = grade 1), gray for "not relevant"
SURFACE = "#1a1a19"  # dark surface, to sit in VS Code's dark theme
INK = "#ffffff"
INK2 = "#c3c2b7"
GRID = "#3a3a38"
G2 = "#3987e5"  # grade 2: strong blue
G1 = "#24507f"  # grade 1: muted blue
G0 = "#2e2e2c"  # not relevant
ACCENT = "#e8763f"

plt.rcParams.update(
    {
        "font.size": 11,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
    }
)

GRADES = [0, 2, 0, 1, 0, 0, 1, 0, 0, 0]  # the worked example: ranks 1..10
FILL = {2: G2, 1: G1, 0: G0}


def strip(ax, y, grades, label=None, x0=0.0, w=0.9, h=0.7):
    """Draw a row of rank boxes, coloured by relevance grade."""
    for i, g in enumerate(grades):
        ax.add_patch(
            FancyBboxPatch(
                (x0 + i, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc=FILL[g], ec="none"
            )
        )
        ax.text(
            x0 + i + w / 2,
            y + h / 2,
            str(g) if g else "–",
            ha="center",
            va="center",
            fontsize=10,
            color=INK if g else INK2,
            fontweight="bold" if g else "normal",
        )
    if label:
        ax.text(x0 - 0.3, y + h / 2, label, ha="right", va="center", fontsize=11, color=INK)


def legend_row(ax, y):
    items = [(G2, "grade 2: contains the full answer"), (G1, "grade 1: contains a needed part"),
             (G0, "not relevant")]
    x = 0.0
    for colour, text in items:
        ax.add_patch(Rectangle((x, y), 0.35, 0.35, fc=colour, ec="none"))
        ax.text(x + 0.5, y + 0.17, text, va="center", fontsize=9.5, color=INK2)
        x += 5.9


def fig_pipeline(path):
    fig, ax = plt.subplots(figsize=(11, 1.7))
    ax.set_xlim(-0.15, 11.2)
    ax.set_ylim(0.25, 1.95)
    ax.axis("off")
    w, gap = 1.9, 0.42
    boxes = [
        ("Question", ""),
        ("Search", "2,313 pages"),
        ("3 pages", "as images"),
        ("Qwen3-VL-4B", "reads them"),
        ("Answer", "+ cited page"),
    ]
    for i, (title, sub) in enumerate(boxes):
        x = i * (w + gap)
        hl = i == 1
        ax.add_patch(FancyBboxPatch((x, 0.9), w, 0.9, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=G0, ec=G2 if hl else GRID, lw=1.8 if hl else 1.0))
        ax.text(x + w / 2, 1.45 if sub else 1.35, title, ha="center", va="center", fontsize=13,
                fontweight="bold", color=INK)
        if sub:
            ax.text(x + w / 2, 1.13, sub, ha="center", va="center", fontsize=10, color=INK2)
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.04, 1.35), (x + w + gap - 0.04, 1.35), arrowstyle="-|>",
                                         mutation_scale=13, color=INK2, lw=1.3))
    x = w + gap
    ax.text(x + w / 2, 0.55, "scored in this notebook", ha="center", fontsize=10, color=G2)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def fig_metrics(path):
    rel_ranks = [i + 1 for i, g in enumerate(GRADES) if g]
    dcg = sum(GRADES[r - 1] / math.log2(r + 1) for r in rel_ranks)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(GRADES, reverse=True)) if g)
    fig, ax = plt.subplots(figsize=(11, 3.1))
    ax.set_xlim(-1.2, 14.6)
    ax.set_ylim(-1.05, 2.35)
    ax.axis("off")
    for i in range(10):
        ax.text(i + 0.45, 1.25, str(i + 1), ha="center", fontsize=9.5, color=INK2)
    ax.text(-0.25, 1.25, "rank", ha="right", fontsize=9.5, color=INK2)
    strip(ax, 0.2, GRADES)
    ax.add_patch(Rectangle((-0.12, 0.05), 5.0, 1.0, fill=False, ec=ACCENT, lw=1.8, ls="--"))
    ax.text(2.45, -0.25, "top 5", ha="center", fontsize=9.5, color=ACCENT)
    ax.text(-1.2, 2.05, "Search returned 10 pages; the 3 right ones are at ranks 2, 4 and 7",
            fontsize=12.5, fontweight="bold", color=INK)
    scores = [("Hit@5", "1"), ("Recall@5", "0.67"), ("MRR@10", "0.50"), ("nDCG@10", f"{dcg / idcg:.2f}")]
    for j, (name, val) in enumerate(scores):
        y = 1.2 - j * 0.45
        ax.text(11.0, y, name, fontsize=11.5, color=INK2, va="center")
        ax.text(14.4, y, val, fontsize=12.5, fontweight="bold", color=INK, va="center", ha="right")
    legend_row(ax, -0.95)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def box(ax, x, y, w, h, title, sub="", hl=False, fs=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc=G0, ec=G2 if hl else GRID, lw=1.8 if hl else 1.0))
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), title, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=INK)
    if sub:
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=9, color=INK2)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=INK2, lw=1.2))


def fig_rag(path):
    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    ax.set_xlim(-1.7, 12.6)
    ax.set_ylim(-0.55, 3.55)
    ax.axis("off")
    w, h = 1.85, 0.95
    # offline row
    ax.text(-1.7, 2.95, "once", fontsize=10.5, color=INK2, va="center")
    box(ax, 0.0, 2.5, w, h, "Pages", "2,313")
    box(ax, 2.4, 2.5, w, h, "Index", "words, vectors", hl=True)
    arrow(ax, 0.0 + w + 0.05, 2.97, 2.35, 2.97)
    # online row
    ax.text(-1.7, 0.95, "per question", fontsize=10.5, color=INK2, va="center")
    row = [(0.0, "Question", ""), (2.4, "Retrieve", "top 20", True), (4.8, "Rerank", "top 3", True),
           (7.2, "Generate", "LLM reads pages"), (9.6, "Answer", "+ cited pages")]
    for i, (x, t, sub, *hl) in enumerate(row):
        box(ax, x, 0.5, w, h, t, sub, hl=bool(hl))
        if i:
            arrow(ax, row[i - 1][0] + w + 0.05, 0.97, x - 0.05, 0.97)
    arrow(ax, 2.4 + w / 2, 2.45, 2.4 + w / 2, 0.5 + h + 0.05)
    ax.text(2.4 + w / 2 + 0.12, 1.95, "search the index", fontsize=9, color=INK2)
    ax.plot([2.4, 4.8 + w], [0.3, 0.3], color=G2, lw=2)
    ax.text(2.4 + (2.4 + w) / 2, 0.02, "this notebook: the R in RAG", ha="center", va="center", fontsize=10,
            color=G2)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def vecs(ax, x, y, n, colour, size=0.22, gap=0.06, vertical=False):
    """n small squares in a row (or column): a set of vectors."""
    for i in range(n):
        dx, dy = (0, -i * (size + gap)) if vertical else (i * (size + gap), 0)
        ax.add_patch(Rectangle((x + dx, y + dy), size, size, fc=colour, ec="none"))


def fig_scoring(path):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 5.6))
    for a in axes.flat:
        a.set_xlim(0, 10)
        a.set_ylim(0, 4.4)
        a.axis("off")

    a = axes[0, 0]
    a.set_title("Keywords (BM25)", loc="left", fontsize=12.5, fontweight="bold", color=INK)
    box(a, 0.2, 2.6, 3.6, 0.9, "question words", fs=10.5)
    box(a, 0.2, 0.8, 3.6, 0.9, "page words", fs=10.5)
    arrow(a, 3.9, 3.05, 5.3, 2.25)
    arrow(a, 3.9, 1.25, 5.3, 2.05)
    box(a, 5.4, 1.65, 3.6, 0.95, "count shared words", "rare words count more", fs=10.5)
    a.text(0.2, 0.2, "Fast, exact terms. Misses synonyms and charts.", fontsize=9.5, color=INK2)

    a = axes[0, 1]
    a.set_title("Dense embeddings (bi-encoder)", loc="left", fontsize=12.5, fontweight="bold", color=INK)
    box(a, 0.2, 2.6, 2.4, 0.9, "question", fs=10.5)
    box(a, 0.2, 0.8, 2.4, 0.9, "page", fs=10.5)
    arrow(a, 2.7, 3.05, 3.5, 3.05)
    arrow(a, 2.7, 1.25, 3.5, 1.25)
    vecs(a, 3.6, 2.95, 1, ACCENT, size=0.28)
    vecs(a, 3.6, 1.12, 1, G2, size=0.28)
    a.text(3.6, 3.45, "one vector", fontsize=9, color=INK2, va="center")
    a.text(3.6, 0.85, "one vector, made in advance", fontsize=9, color=INK2, va="center")
    box(a, 6.8, 1.65, 2.6, 0.95, "similarity", "one dot product", fs=10.5)
    arrow(a, 3.95, 2.95, 6.75, 2.3)
    arrow(a, 3.95, 1.4, 6.75, 2.0)
    a.text(0.2, 0.2, "Matches meaning, fast at scale. One vector blurs details.", fontsize=9.5, color=INK2)

    a = axes[1, 0]
    a.set_title("Late interaction (ColBERT, ColPali / ColQwen)", loc="left", fontsize=12.5, fontweight="bold",
                color=INK)
    box(a, 0.2, 2.6, 2.4, 0.9, "question", fs=10.5)
    box(a, 0.2, 0.8, 2.4, 0.9, "page image", fs=10.5)
    arrow(a, 2.7, 3.05, 3.5, 3.05)
    arrow(a, 2.7, 1.25, 3.5, 1.25)
    vecs(a, 3.6, 2.92, 5, ACCENT)
    vecs(a, 3.6, 1.12, 11, G2, size=0.2, gap=0.04)
    a.text(3.6, 3.35, "a vector per word", fontsize=9, color=INK2)
    a.text(3.6, 0.82, "a vector per image patch, made in advance", fontsize=9, color=INK2)
    box(a, 6.5, 1.75, 3.4, 0.95, "MaxSim", "best matches, summed", fs=10.5)
    arrow(a, 5.0, 3.0, 6.9, 2.75)
    arrow(a, 6.6, 1.3, 7.2, 1.7)
    a.text(0.2, 0.2, "Keeps details, reads layout and charts. Stores many vectors per page.", fontsize=9.5,
           color=INK2)

    a = axes[1, 1]
    a.set_title("Cross-encoder (reranker)", loc="left", fontsize=12.5, fontweight="bold", color=INK)
    box(a, 0.2, 1.7, 3.2, 0.95, "question + page", "read together", fs=10.5)
    arrow(a, 3.5, 2.17, 4.4, 2.17)
    box(a, 4.5, 1.7, 2.0, 0.95, "model", fs=10.5)
    arrow(a, 6.6, 2.17, 7.4, 2.17)
    box(a, 7.5, 1.7, 1.9, 0.95, "score", fs=10.5)
    a.text(0.2, 0.2, "Most accurate. Slow: one model call per page, so only for the top 20.", fontsize=9.5,
           color=INK2)

    fig.tight_layout(h_pad=1.5, w_pad=2.0)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def fig_failures(path):
    fig, ax = plt.subplots(figsize=(11.5, 3.0))
    ax.set_xlim(-0.15, 12.0)
    ax.set_ylim(-0.6, 1.95)
    ax.axis("off")
    w, gap = 2.0, 0.45
    steps = [("Question", ""), ("Search", ""), ("Top pages", "as images"), ("Qwen3-VL-4B", "reads and writes"),
             ("Answer", "+ cited pages")]
    for i, (t, sub) in enumerate(steps):
        x = i * (w + gap)
        box(ax, x, 0.8, w, 0.95, t, sub)
        if i:
            arrow(ax, x - gap + 0.04, 1.27, x - 0.04, 1.27)
    fails = {1: ["1  right page not found"], 3: ["2  page misread", "3  fact made up"],
             4: ["4  wrong page cited", "5  should have abstained"]}
    for i, lines in fails.items():
        x = i * (w + gap) + w / 2
        ax.plot([x, x], [0.75, 0.35], color=ACCENT, lw=1.4)
        for j, line in enumerate(lines):
            ax.text(x, 0.12 - j * 0.42, line, ha="center", va="center", fontsize=10.5, color=ACCENT)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline(HERE / "pipeline_v2.png")
    fig_metrics(HERE / "metrics_v2.png")
    fig_rag(HERE / "rag_overview.png")
    fig_scoring(HERE / "scoring_styles.png")
    fig_failures(HERE / "answer_failures.png")
    print("wrote all figures")
