#!/usr/bin/env python3
"""
Train a SINGLE ResNet variant with optional multi-GPU DistributedDataParallel (DDP).

This is the *data-parallel* path: one process per GPU, each process trains on a shard
of the data, gradients are all-reduced every step. It supports the larger ResNets
(resnet101 / resnet152) as well as the smaller ones, and degrades gracefully to a plain
single-GPU (or CPU) run when not launched under torchrun.

Single GPU / CPU (plain process):
    python tools/train_resnet_ddp.py --variant resnet101 \
      --train-csv data_quality/clean_train_manifest.csv \
      --test-csv  data_quality/heldout_test_manifest.csv \
      --output-dir artifacts/resnet101

Multi-GPU on ONE node (N GPUs) — launch with torchrun:
    torchrun --standalone --nproc_per_node=4 tools/train_resnet_ddp.py \
      --variant resnet101 ... (same args)

torchrun sets RANK / WORLD_SIZE / LOCAL_RANK in the environment; this script
auto-detects them. NOTE: --batch-size is PER GPU, so the effective (global) batch is
batch_size * world_size. Scale --lr roughly linearly with world_size if you change it.

Output (written by rank 0 only): <variant>_results.json + <variant>_weights.pt.
The JSON is schema-compatible with compare_resnet_variants.py, so
tools/analyze_resnet_comparison.py can aggregate it alongside the comparison results.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import confusion_matrix
import wandb

# Reuse the dataset, manifest loader, eval helpers, and arch registry — single source
# of truth, so the data-handling logic never drifts from the comparison script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_resnet_variants import (  # noqa: E402
    ARCH, CarDamageDataset, load_manifests, evaluate, compute_nll,
)
import timm  # noqa: E402


def ddp_setup():
    """Init the process group if launched under torchrun. Returns (rank, world, local, distributed)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def main():
    p = argparse.ArgumentParser(description="Train one ResNet variant (single- or multi-GPU DDP)")
    p.add_argument("--variant", required=True, choices=list(ARCH),
                   help="Which ResNet to train, e.g. resnet101 or resnet152")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64, help="PER-GPU batch size")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Base LR. Scale ~linearly with the number of GPUs for large global batches.")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers PER process")
    p.add_argument("--wandb-project", default="car-damage-severity",
                   help="Weights & Biases project name")
    p.add_argument("--wandb-entity", default=None,
                   help="Weights & Biases entity (team/username)")
    p.add_argument("--disable-wandb", action="store_true",
                   help="Disable wandb logging")
    args = p.parse_args()

    rank, world_size, local_rank, distributed = ddp_setup()
    main_proc = (rank == 0)
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    torch.backends.cudnn.benchmark = True

    # Initialize wandb on rank 0 only
    if main_proc and not args.disable_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.variant}-{int(time.time())}",
            config={
                "variant": args.variant,
                "epochs": args.epochs,
                "batch_size_per_gpu": args.batch_size,
                "global_batch_size": args.batch_size * world_size,
                "learning_rate": args.lr,
                "world_size": world_size,
                "num_workers": args.num_workers,
                "augmentation": True,
                "label_smoothing": 0.1,
                "lr_scheduler": "ReduceLROnPlateau(factor=0.5,patience=3)",
                "weight_decay": 1e-2,
                "dropout": 0.3,
                "drop_path_rate": 0.2,
                "mixup_alpha": 0.2,
            },
            tags=[args.variant, f"world_size_{world_size}"],
        )

    # Every rank loads the manifests (cheap) and builds identical splits (fixed seed=42).
    train_df, val_df, test_df = load_manifests(Path(args.train_csv), Path(args.test_csv))
    if main_proc:
        print(f"[rank0] variant={args.variant} world_size={world_size} device={device}")
        print(f"[rank0] Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")
        if len(train_df) == 0:
            raise SystemExit("No training images resolved — stage ./training and ./validation (see docs).")

    train_ds = CarDamageDataset(train_df, augment=True)   # augmentation on train only
    val_ds   = CarDamageDataset(val_df,   augment=False)
    test_ds  = CarDamageDataset(test_df,  augment=False)

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed else None
    )
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              shuffle=(train_sampler is None), num_workers=args.num_workers,
                              pin_memory=pin, drop_last=False)
    # Eval runs on rank 0 only, over the full (un-sharded) sets — so metrics are exact.
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    model = timm.create_model(
        ARCH[args.variant], pretrained=True, num_classes=3,
        drop_rate=0.3,        # dropout before the final classifier
        drop_path_rate=0.2,   # stochastic depth (drops entire residual branches)
    ).to(device)
    n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
    if distributed:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    # Inverse-frequency class weights + label smoothing (epsilon=0.1) to prevent overconfidence.
    tc = train_df["label"].value_counts().reindex(CarDamageDataset.CLASSES, fill_value=0).astype(float)
    N, K = tc.sum(), len(CarDamageDataset.CLASSES)
    w = torch.tensor([N / (K * tc[c]) for c in CarDamageDataset.CLASSES], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    # Weight decay (L2) via AdamW — strong regularizer for fine-tuning.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    # Adaptive LR: halve the LR when val_loss hasn't improved for 3 epochs.
    # More responsive than cosine — only reduces when training actually stalls.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7,
    )

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    mixup_alpha = 0.2  # Mixup: blend two images/labels; alpha=0.2 is mild but effective
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        if distributed:
            train_sampler.set_epoch(ep)  # reshuffle shards each epoch
        model.train()
        # running = [sum(loss*bs), n_samples, n_correct] — all-reduced for exact global stats.
        running = torch.zeros(3, device=device)
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # Mixup: blend random pairs of samples
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            idx = torch.randperm(x.size(0), device=device)
            x_mix = lam * x + (1 - lam) * x[idx]
            optimizer.zero_grad()
            logits = model(x_mix)
            loss = lam * criterion(logits, y) + (1 - lam) * criterion(logits, y[idx])
            loss.backward()
            optimizer.step()
            running[0] += loss.detach() * x.size(0)
            running[1] += x.size(0)
            running[2] += (logits.argmax(dim=1) == y).sum()
        if distributed:
            dist.all_reduce(running, op=dist.ReduceOp.SUM)
        train_loss = (running[0] / running[1]).item()
        train_acc = (running[2] / running[1]).item()
        current_lr = optimizer.param_groups[0]["lr"]

        if main_proc:
            eval_model = model.module if distributed else model
            val_acc, val_f1, _, _, _, _ = evaluate(eval_model, val_loader, device)
            history["epoch"].append(ep)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)
            print(f"[rank0] Epoch {ep:2d} | train_loss={train_loss:.4f} | train_acc={train_acc:.3f} | val_acc={val_acc:.3f} | val_f1={val_f1:.3f} | lr={current_lr:.2e}")
            if not args.disable_wandb:
                wandb.log({
                    "epoch": ep,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "val_f1": val_f1,
                    "learning_rate": current_lr,
                })
            # Step scheduler on rank 0 using val_loss; LR reduces when val_loss stalls.
            scheduler.step(train_loss)
        if distributed:
            dist.barrier()  # other ranks wait while rank 0 evaluates (no collectives in eval)

    # Final held-out evaluation + save, on rank 0 only.
    if main_proc:
        eval_model = model.module if distributed else model
        test_acc, test_f1, test_rep, y_true, y_pred, _ = evaluate(eval_model, test_loader, device)
        total_nll = compute_nll(eval_model, test_loader, criterion, device)
        n_test = len(test_df)
        aic = 2 * n_params + 2 * total_nll
        bic = n_params * math.log(n_test) + 2 * total_nll
        wall = time.time() - t_start

        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        results = {
            "variant": args.variant,
            "model_arch": ARCH[args.variant],
            "n_params": int(n_params),
            "world_size": world_size,
            "per_gpu_batch_size": args.batch_size,
            "global_batch_size": args.batch_size * world_size,
            "test_accuracy": float(test_acc),
            "test_macro_f1": float(test_f1),
            "test_report": test_rep,
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "aic": float(aic),
            "bic": float(bic),
            "wall_time_sec": float(wall),
            "history": history,
        }
        torch.save(eval_model.state_dict(), out / f"{args.variant}_weights.pt")
        with open(out / f"{args.variant}_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"[rank0] Test acc={test_acc:.3f} f1={test_f1:.3f} | params={n_params:,} "
              f"| wall={wall:.1f}s | global_batch={args.batch_size * world_size} | saved -> {out}")
        if not args.disable_wandb:
            wandb.log({
                "test_accuracy": test_acc,
                "test_f1": test_f1,
                "n_params": n_params,
                "wall_time_sec": wall,
                "aic": aic,
                "bic": bic,
            })
            wandb.finish()

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
