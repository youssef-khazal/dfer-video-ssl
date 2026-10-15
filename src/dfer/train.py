"""Fine-tune one video backbone on one DFEW fold, then evaluate it once on the fold's test set.

Example:
    python -m dfer.train --dfew-root "data/DFEW" --model videomae_k400 --fold 1 --amp

Model selection (best epoch, early stopping) uses a validation split taken from the fold's
training data, never the test set. Outputs in ``--out-dir/<model>_fold<k>_seed<s>/``:

    best.pth        weights with the best validation UAR
    last.pth        full training state, used by --resume after an interruption
    history.csv     per-epoch train loss / val loss / val WAR / val UAR
    metrics.json    validation and test WAR/UAR, per-class test recall, confusion matrix, config
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from dfer.data import EMOTIONS, NUM_CLASSES, ClipTransform, DFEWClips, class_weights, fold_tables
from dfer.metrics import per_class_recall, uar, war
from dfer.models import MODELS, VideoClassifier, build_model, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model: VideoClassifier, loader: DataLoader, device: torch.device, criterion: nn.Module | None = None, amp: bool = False):
    """Return (y_true, y_pred, mean_loss) over a loader."""
    model.eval()
    y_true, y_pred, total_loss = [], [], 0.0
    for video, label in tqdm(loader, desc="eval", leave=False):
        video, label = video.to(device, non_blocking=True), label.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(video)
        if criterion is not None:
            total_loss += criterion(logits.float(), label).item() * len(label)
        y_true.append(label.cpu().numpy())
        y_pred.append(logits.argmax(1).cpu().numpy())
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    return y_true, y_pred, total_loss / len(y_true)


def make_stages(args: argparse.Namespace) -> list[dict]:
    """Full fine-tuning = one stage. Two-stage = head only, then head + last blocks."""
    if not MODELS[args.model].two_stage:
        return [{"name": "full", "epochs": args.epochs, "last_blocks": None, "lr_head": args.lr, "lr_body": args.lr}]
    return [
        {"name": "head", "epochs": args.stage1_epochs, "last_blocks": 0, "lr_head": args.lr_head, "lr_body": 0.0},
        {"name": "partial", "epochs": args.epochs, "last_blocks": args.unfreeze_blocks, "lr_head": args.lr_head * 0.5, "lr_body": args.lr},
    ]


def make_optimizer(model: VideoClassifier, stage: dict, weight_decay: float) -> torch.optim.Optimizer:
    model.set_trainable(stage["last_blocks"])
    head_ids = {id(p) for p in model.head().parameters()}
    head = [p for p in model.parameters() if p.requires_grad and id(p) in head_ids]
    body = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": head, "lr": stage["lr_head"]}]
    if body:
        groups.append({"params": body, "lr": stage["lr_body"]})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def write_history(history: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.amp and device.type == "cuda"
    run_dir = args.out_dir / f"{args.model}_fold{args.fold}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = fold_tables(args.dfew_root, args.fold, args.val_fraction, args.seed)
    clip_root = Path(args.dfew_root) / "Clip" / "clip_224x224_16f"
    size = args.img_size
    loader = lambda df, train_mode: DataLoader(  # noqa: E731
        DFEWClips(clip_root, df, ClipTransform(size, train=train_mode), args.num_frames),
        batch_size=args.batch_size, shuffle=train_mode, num_workers=args.workers, pin_memory=True,
        drop_last=train_mode, persistent_workers=args.workers > 0,
    )
    train_loader, val_loader, test_loader = loader(train_df, True), loader(val_df, False), loader(test_df, False)
    print(f"Device {device} | train {len(train_df)} | val {len(val_df)} | test {len(test_df)} clips")

    model = build_model(args.model, NUM_CLASSES).to(device)
    if args.grad_checkpointing:  # trades ~30% speed for much less GPU memory
        model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_df["label"]).to(device))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    stages = make_stages(args)

    history, best_uar, start_stage, start_epoch = [], -1.0, 0, 0
    last_path = run_dir / "last.pth"
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        history, best_uar = state["history"], state["best_uar"]
        start_stage, start_epoch = state["stage"], state["epoch"] + 1
        print(f"Resumed at stage {start_stage}, epoch {start_epoch + 1}, best val UAR {best_uar:.4f}")

    for s, stage in enumerate(stages):
        if s < start_stage:
            continue
        optimizer = make_optimizer(model, stage, args.weight_decay)
        steps_per_epoch = len(train_loader) // args.grad_accum
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage["epochs"] * steps_per_epoch, eta_min=1e-7)
        first_epoch, wait = 0, 0
        if s == start_stage and start_epoch > 0:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            first_epoch, wait = start_epoch, state["wait"]
        print(f"Stage '{stage['name']}': {count_parameters(model, trainable_only=True) / 1e6:.1f}M trainable parameters")

        for epoch in range(first_epoch, stage["epochs"]):
            t0 = time.time()
            model.train()
            running, seen = 0.0, 0
            optimizer.zero_grad(set_to_none=True)
            for step, (video, label) in enumerate(tqdm(train_loader, desc=f"{stage['name']} {epoch + 1}/{stage['epochs']}", leave=False), 1):
                video, label = video.to(device, non_blocking=True), label.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    loss = criterion(model(video).float(), label)
                scaler.scale(loss / args.grad_accum).backward()
                if step % args.grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                running += loss.item() * len(label)
                seen += len(label)

            y_true, y_pred, val_loss = predict(model, val_loader, device, criterion, amp)
            row = {"stage": stage["name"], "epoch": epoch + 1, "train_loss": running / seen, "val_loss": val_loss,
                   "val_war": war(y_true, y_pred), "val_uar": uar(y_true, y_pred, NUM_CLASSES),
                   "minutes": (time.time() - t0) / 60}
            history.append(row)
            write_history(history, run_dir / "history.csv")
            print(f"[{stage['name']}] epoch {epoch + 1:02d} | {row['minutes']:.1f} min | train loss {row['train_loss']:.3f} | "
                  f"val loss {val_loss:.3f} | val WAR {row['val_war']:.3f} | val UAR {row['val_uar']:.3f}")

            if row["val_uar"] > best_uar + 1e-4:
                best_uar, wait = row["val_uar"], 0
                torch.save(model.state_dict(), run_dir / "best.pth")
            else:
                wait += 1
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(), "stage": s, "epoch": epoch, "wait": wait,
                        "history": history, "best_uar": best_uar}, last_path)
            if wait >= args.patience:
                print("Early stopping this stage.")
                break
        start_epoch = 0
        if s + 1 < len(stages):  # continue the next stage from the best weights so far
            model.load_state_dict(torch.load(run_dir / "best.pth", map_location=device, weights_only=True))

    # ---- final, single evaluation on the untouched test set ----
    model.load_state_dict(torch.load(run_dir / "best.pth", map_location=device, weights_only=True))
    v_true, v_pred, _ = predict(model, val_loader, device, amp=amp)
    t_true, t_pred, _ = predict(model, test_loader, device, amp=amp)
    best_row = max(history, key=lambda r: r["val_uar"])
    metrics = {
        "model": args.model, "checkpoint": MODELS[args.model].checkpoint, "fold": args.fold,
        "best_stage": best_row["stage"], "best_epoch": best_row["epoch"],
        "val": {"war": war(v_true, v_pred), "uar": uar(v_true, v_pred, NUM_CLASSES)},
        "test": {"war": war(t_true, t_pred), "uar": uar(t_true, t_pred, NUM_CLASSES)},
        "test_recall_per_class": dict(zip(EMOTIONS, np.nan_to_num(per_class_recall(t_true, t_pred, NUM_CLASSES)).round(4).tolist())),
        "test_confusion_matrix": confusion_matrix(t_true, t_pred, labels=list(range(NUM_CLASSES))).tolist(),
        "train_minutes": round(sum(r["minutes"] for r in history), 1),
        "parameters_m": round(count_parameters(model) / 1e6, 1),
        "config": {k: str(v) for k, v in vars(args).items()},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"TEST  WAR {metrics['test']['war']:.4f} | UAR {metrics['test']['uar']:.4f}  ->  {run_dir}")
    if not args.keep_last:
        last_path.unlink(missing_ok=True)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dfew-root", type=Path, required=True)
    p.add_argument("--model", choices=sorted(MODELS), default="videomae_k400")
    p.add_argument("--fold", type=int, default=1, choices=range(1, 6))
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=12, help="Epochs of full / partial fine-tuning")
    p.add_argument("--stage1-epochs", type=int, default=4, help="Head-only epochs (two-stage models)")
    p.add_argument("--unfreeze-blocks", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-5, help="Backbone learning rate")
    p.add_argument("--lr-head", type=float, default=1e-3, help="Head learning rate in two-stage training")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4, help="Effective batch = batch-size x grad-accum")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", help="Mixed precision on GPU")
    p.add_argument("--grad-checkpointing", action="store_true", help="Lower GPU memory use (slower)")
    p.add_argument("--resume", action="store_true", help="Continue from last.pth after an interruption")
    p.add_argument("--keep-last", action="store_true", help="Keep last.pth after training")
    p.add_argument("--out-dir", type=Path, default=Path("runs"))
    return p.parse_args(argv)


def train_with_oom_fallback(args: argparse.Namespace) -> dict:
    """Train; on GPU out-of-memory, halve the batch and double gradient accumulation.

    The effective batch (batch_size x grad_accum) stays the same, and these models use
    LayerNorm (no batch-size-dependent layers), so the training setup stays comparable.
    Training resumes from the last finished epoch.
    """
    while True:
        try:
            return train(args)
        except torch.OutOfMemoryError:
            if args.batch_size <= 1:
                raise
            args.batch_size //= 2
            args.grad_accum *= 2
            args.resume = True
            torch.cuda.empty_cache()
            print(f"GPU out of memory -> retrying with batch {args.batch_size} x accumulation {args.grad_accum} "
                  f"(same effective batch of {args.batch_size * args.grad_accum})", flush=True)


if __name__ == "__main__":
    train_with_oom_fallback(parse_args())
