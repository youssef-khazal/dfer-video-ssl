"""Collect finished runs into a results table, per-class recall table and figures.

    python -m dfer.report --runs-dir runs --out-dir results
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dfer.data import EMOTIONS  # noqa: E402

PRETTY = {
    "videomae_k400": "VideoMAE-B, K400 supervised (full FT)",
    "videomae_ssl": "VideoMAE-B, self-supervised (full FT)",
    "videomae_ssl_2stage": "VideoMAE-B, self-supervised (2-stage FT)",
    "vjepa2": "V-JEPA 2 ViT-L, self-supervised (2-stage FT)",
}


def load_runs(runs_dir: Path) -> list[dict]:
    runs = []
    for f in sorted(runs_dir.glob("*/metrics.json")):
        m = json.loads(f.read_text())
        with open(f.parent / "history.csv", newline="") as h:
            m["history"] = list(csv.DictReader(h))
        runs.append(m)
    return sorted(runs, key=lambda m: -m["test"]["uar"])


def results_table(runs: list[dict]) -> str:
    lines = ["| Model | Params | Test UAR | Test WAR | Val UAR | Train time |", "|---|---|---|---|---|---|"]
    for m in runs:
        lines.append(f"| {PRETTY.get(m['model'], m['model'])} | {m['parameters_m']:.0f}M | **{m['test']['uar']:.1%}** | "
                     f"{m['test']['war']:.1%} | {m['val']['uar']:.1%} | {m['train_minutes'] / 60:.1f} h |")
    return "\n".join(lines)


def recall_table(runs: list[dict]) -> str:
    lines = ["| Model | " + " | ".join(EMOTIONS) + " |", "|---|" + "---|" * len(EMOTIONS)]
    for m in runs:
        r = m["test_recall_per_class"]
        lines.append(f"| {PRETTY.get(m['model'], m['model'])} | " + " | ".join(f"{r[e]:.0%}" for e in EMOTIONS) + " |")
    return "\n".join(lines)


# Fixed categorical order (one colour per model, never re-assigned) and a single-hue ramp for matrices.
COLORS = {"videomae_k400": "#2a78d6", "videomae_ssl": "#eb6834", "videomae_ssl_2stage": "#1baf7a", "vjepa2": "#6250d6"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b8b7b1")
    ax.tick_params(colors=MUTED)


def plot_confusion(m: dict, path: Path) -> None:
    cm = np.array(m["test_confusion_matrix"], dtype=float)
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.6, 4.9), facecolor=SURFACE)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > 0.5 else INK)
    ax.set_xticks(range(len(EMOTIONS)), EMOTIONS, rotation=45, ha="right")
    ax.set_yticks(range(len(EMOTIONS)), EMOTIONS)
    ax.set_xlabel("Predicted", color=MUTED)
    ax.set_ylabel("True", color=MUTED)
    ax.set_title(f"{PRETTY.get(m['model'], m['model'])}\nrow-normalised confusion matrix (test set)", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_curves(runs: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4), facecolor=SURFACE)
    for m in sorted(runs, key=lambda r: list(COLORS).index(r["model"]) if r["model"] in COLORS else 99):
        uars = [float(r["val_uar"]) * 100 for r in m["history"]]
        ax.plot(range(1, len(uars) + 1), uars, marker="o", ms=4, linewidth=2,
                color=COLORS.get(m["model"]), label=PRETTY.get(m["model"], m["model"]))
    ax.set_xlabel("Epoch (head-only epochs included for 2-stage models)", color=MUTED)
    ax.set_ylabel("Validation UAR (%)", color=MUTED)
    ax.set_title("Validation UAR during fine-tuning", loc="left", color=INK, fontsize=11)
    _style(ax)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_recall(runs: list[dict], path: Path) -> None:
    """Per-emotion test recall, one bar per model (shows where the rare classes fail)."""
    ordered = sorted(runs, key=lambda r: list(COLORS).index(r["model"]) if r["model"] in COLORS else 99)
    x = np.arange(len(EMOTIONS))
    width = 0.8 / len(ordered)
    fig, ax = plt.subplots(figsize=(8.5, 3.8), facecolor=SURFACE)
    for k, m in enumerate(ordered):
        r = [m["test_recall_per_class"][e] * 100 for e in EMOTIONS]
        ax.bar(x + (k - (len(ordered) - 1) / 2) * width, r, width * 0.92, color=COLORS.get(m["model"]),
               label=PRETTY.get(m["model"], m["model"]))
    ax.set_xticks(x, EMOTIONS)
    ax.set_ylabel("Test recall (%)", color=MUTED)
    ax.set_ylim(0, 100)
    ax.set_title("Recall per emotion on the test set", loc="left", color=INK, fontsize=11)
    _style(ax)
    ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    args = p.parse_args()
    runs = load_runs(args.runs_dir)
    if not runs:
        raise SystemExit(f"No finished runs in {args.runs_dir}")
    figs = args.out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    text = results_table(runs) + "\n\n" + recall_table(runs) + "\n"
    (args.out_dir / "results.md").write_text(text)
    plot_curves(runs, figs / "val_uar_curves.png")
    plot_recall(runs, figs / "recall_per_emotion.png")
    plot_confusion(runs[0], figs / "confusion_best_model.png")
    print(text)


if __name__ == "__main__":
    main()
