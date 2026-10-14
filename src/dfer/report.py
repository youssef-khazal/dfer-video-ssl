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


def plot_confusion(m: dict, path: Path) -> None:
    cm = np.array(m["test_confusion_matrix"], dtype=float)
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(EMOTIONS)), EMOTIONS, rotation=45, ha="right")
    ax.set_yticks(range(len(EMOTIONS)), EMOTIONS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{PRETTY.get(m['model'], m['model'])}\nnormalised confusion matrix (test)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(runs: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for m in runs:
        uars = [float(r["val_uar"]) for r in m["history"]]
        ax.plot(range(1, len(uars) + 1), uars, marker="o", ms=3, label=PRETTY.get(m["model"], m["model"]))
    ax.set(xlabel="Epoch (all stages)", ylabel="Validation UAR", title="Validation UAR during fine-tuning")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
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
    plot_confusion(runs[0], figs / "confusion_best_model.png")
    print(text)


if __name__ == "__main__":
    main()
