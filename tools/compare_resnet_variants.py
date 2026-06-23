#!/usr/bin/env python3
"""
Compare ResNet variants on car damage severity classification (default: ResNet152).

Trains each variant and logs:
  - Training curves (loss, accuracy, F1)
  - Test metrics (accuracy, macro-F1, per-class precision/recall/F1)
  - Model size (parameters, disk MB)
  - Inference time
  - AIC/BIC for model selection

Usage:
  python tools/compare_resnet_variants.py \\
    --data-dir data/images \\
    --labels-csv data_quality/clean_train_manifest.csv \\
    --test-csv data_quality/heldout_test_manifest.csv \\
    --output-dir artifacts/resnet_comparison \\
    --epochs 8 \\
    --batch-size 32
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import timm


class CarDamageDataset(Dataset):
    """Image classification dataset with optional train-time augmentation."""
    CLASSES = ["minor", "moderate", "severe"]
    LBL = {c: i for i, c in enumerate(CLASSES)}
    IMSIZE = 224
    MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
    STD  = np.array([0.229, 0.224, 0.225], dtype="float32")
    IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")

        if self.augment:
            # Random resized crop: randomly zooms in/out before taking 224×224 patch
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img, scale=(0.7, 1.0), ratio=(3/4, 4/3)
            )
            img = transforms.functional.resized_crop(img, i, j, h, w, (self.IMSIZE, self.IMSIZE))
            # Horizontal flip
            if np.random.rand() < 0.5:
                img = transforms.functional.hflip(img)
            # Color jitter: brightness, contrast, saturation, hue
            img = transforms.functional.adjust_brightness(img, 1 + np.random.uniform(-0.3, 0.3))
            img = transforms.functional.adjust_contrast(img,  1 + np.random.uniform(-0.3, 0.3))
            img = transforms.functional.adjust_saturation(img, 1 + np.random.uniform(-0.2, 0.2))
            # Random rotation ±15°
            angle = np.random.uniform(-15, 15)
            img = transforms.functional.rotate(img, angle)
        else:
            img = img.resize((self.IMSIZE, self.IMSIZE))

        arr = np.asarray(img).astype("float32") / 255.0
        arr = (arr - self.MEAN) / self.STD
        arr = arr.transpose(2, 0, 1)
        x = torch.from_numpy(arr.copy())
        y = self.LBL[row["label"]]
        return x, y


def load_manifests(train_csv: Path, test_csv: Path) -> tuple:
    """Load and prepare train/val/test splits."""
    train_df = pd.read_csv(train_csv).rename(columns={"file_path": "path"})
    test_df = pd.read_csv(test_csv).rename(columns={"file_path": "path"})

    # Filter to valid images and resolve paths
    def resolve_path(p, source_label_dir=None):
        """Try to find a valid image file, returning resolved path or None."""
        p_path = Path(p)
        # Try original path first
        if p_path.exists():
            return str(p_path)

        # Try to extract filename and search in local training/validation dirs
        filename = p_path.name
        label_dir = None

        if source_label_dir and isinstance(source_label_dir, str):
            # Use just the last path component in case it's a full path
            label_dir = Path(source_label_dir).name.lower()

        # Try to infer label from path if source_label_dir is not available
        if label_dir is None:
            for part in p_path.parts:
                if "minor" in part.lower():
                    label_dir = "minor"
                    break
                elif "moderate" in part.lower():
                    label_dir = "moderate"
                    break
                elif "severe" in part.lower():
                    label_dir = "severe"
                    break

        # Search in local training/validation dirs (with or without a 'data/' prefix)
        for split_dir in [
            Path("training"), Path("validation"),
            Path("data") / "training", Path("data") / "validation",
        ]:
            if not split_dir.exists():
                continue
            for class_dir in split_dir.glob("*"):
                if label_dir and label_dir not in class_dir.name.lower():
                    continue
                candidate = class_dir / filename
                if candidate.exists():
                    return str(candidate)

        return None

    if "source_label_dir" in train_df.columns:
        train_df["path"] = train_df.apply(lambda r: resolve_path(r["path"], r["source_label_dir"]), axis=1)
    else:
        train_df["path"] = train_df["path"].map(resolve_path)
    train_df = train_df.dropna(subset=["path"]).reset_index(drop=True)
    print(f"[load_manifests] train resolved {len(train_df)} images (cwd={Path.cwd()})")

    if "source_label_dir" in test_df.columns:
        test_df["path"] = test_df.apply(lambda r: resolve_path(r["path"], r["source_label_dir"]), axis=1)
    else:
        test_df["path"] = test_df["path"].map(resolve_path)
    test_df = test_df.dropna(subset=["path"]).reset_index(drop=True)
    print(f"[load_manifests] test resolved {len(test_df)} images")

    # Internal train/val split
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, stratify=train_df["label"], random_state=42
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model on loader."""
    model.eval()
    y_true, y_pred, y_proba = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        y_pred.extend(torch.argmax(logits, 1).cpu().numpy())
        y_proba.extend(torch.softmax(logits, 1).cpu().numpy())
        y_true.extend(y.numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)
    
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    rep = classification_report(y_true, y_pred, 
                                target_names=CarDamageDataset.CLASSES, 
                                output_dict=True)
    return acc, f1m, rep, y_true, y_pred, y_proba


@torch.no_grad()
def compute_nll(model, loader, criterion, device):
    """Compute total NLL for AIC/BIC."""
    model.eval()
    total_nll = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        nll = criterion(logits, y)
        total_nll += nll.item() * x.size(0)
    return total_nll


def train_variant(variant_name: str, model_arch: str, train_df, val_df, test_df,
                  epochs: int, batch_size: int, device: str, output_dir: Path):
    """Train a single ResNet variant and save results."""
    
    print(f"\n{'='*60}")
    print(f"Training {variant_name} ({model_arch})")
    print(f"{'='*60}")
    
    # Dataloaders
    train_ds = CarDamageDataset(train_df)
    val_ds = CarDamageDataset(val_df)
    test_ds = CarDamageDataset(test_df)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    
    # Model
    model = timm.create_model(model_arch, pretrained=True, num_classes=3).to(device)
    
    # Class weights
    tc = train_df["label"].value_counts().reindex(CarDamageDataset.CLASSES, fill_value=0).astype(float)
    N = tc.sum()
    K = len(CarDamageDataset.CLASSES)
    w = torch.tensor([N / (K * tc[c]) for c in CarDamageDataset.CLASSES], 
                     dtype=torch.float32, device=device)
    
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Count params
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Training loop
    history = {"epoch": [], "train_loss": [], "val_acc": [], "val_f1": []}
    best_val_f1 = -1.0
    
    for ep in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, val_f1, _, _, _, _ = evaluate(model, val_loader, device)
        history["epoch"].append(ep)
        history["train_loss"].append(tr_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        print(f"  Epoch {ep:2d} | train_loss={tr_loss:.4f} | val_acc={val_acc:.3f} | val_f1={val_f1:.3f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
    
    # Evaluation on held-out test
    test_acc, test_f1, test_rep, y_true, y_pred, y_proba = evaluate(model, test_loader, device)
    
    # AIC/BIC
    total_nll = compute_nll(model, test_loader, criterion, device)
    n_test = len(test_df)
    aic = 2 * n_params + 2 * total_nll
    bic = n_params * math.log(n_test) + 2 * total_nll
    
    # Inference time (100 samples)
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        for i, (x, y) in enumerate(test_loader):
            if i >= 2:  # ~128 samples
                break
            x = x.to(device)
            _ = model(x)
        infer_time = (time.time() - t0) / 128  # ms per sample
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    results = {
        "variant": variant_name,
        "model_arch": model_arch,
        "n_params": int(n_params),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1),
        "test_report": test_rep,
        "confusion_matrix": cm.tolist(),
        "aic": float(aic),
        "bic": float(bic),
        "inference_ms_per_sample": float(infer_time),
        "history": history,
    }
    
    # Save results
    result_file = output_dir / f"{variant_name}_results.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Test Accuracy: {test_acc:.3f}")
    print(f"  Test Macro F1: {test_f1:.3f}")
    print(f"  Parameters: {n_params:,}")
    print(f"  AIC: {aic:.2f} | BIC: {bic:.2f}")
    print(f"  Inference: {infer_time:.4f} ms/sample")
    print(f"  Saved to: {result_file}")
    
    return results


# Known architectures: variant name -> timm model id.
# The 18/34/50 tags are the proven "ResNet Strikes Back" A1 weights. For the larger
# 101/152 models we use the bare timm model name, which resolves to timm's default
# pretrained tag — robust against a specific tag being unavailable (avoids a Hub 404).
ARCH = {
    "resnet18":          "resnet18.a1_in1k",
    "resnet34":          "resnet34.a1_in1k",
    "resnet50":          "resnet50.a1_in1k",
    "resnet101":         "resnet101",
    "resnet152":         "resnet152",
    # Deeper / wider variants — more capacity, still fit on one H200.
    "resnet200d":        "resnet200d",          # 65M params, deeper stem
    "wide_resnet101_2":  "wide_resnet101_2",    # 127M params, 2× channel width
    "resnext101_32x8d": "resnext101_32x8d",    # 89M params, grouped convolutions
    "resnext101_64x4d": "resnext101_64x4d",    # 84M params
}

# Default comparison set: resnext101_32x8d (grouped convolutions, stronger than resnet152).
DEFAULT_VARIANTS = ["resnext101_32x8d"]


def main():
    parser = argparse.ArgumentParser(description="Compare ResNet variants")
    parser.add_argument("--train-csv", required=True, help="Path to training manifest CSV")
    parser.add_argument("--test-csv", required=True, help="Path to held-out test CSV")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                        choices=list(ARCH),
                        help="Which ResNet variants to train (default: resnet152). "
                             "Other models resnet18 / resnet34 / resnet50 / resnet101 are also available. "
                             "Pass multiple to compare: --variants resnet18 resnet50 resnet152")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_df, val_df, test_df = load_manifests(Path(args.train_csv), Path(args.test_csv))
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # ResNet variants to compare (selected via --variants)
    variants = [(name, ARCH[name]) for name in args.variants]
    
    all_results = {}
    for name, arch in variants:
        results = train_variant(
            name, arch, train_df, val_df, test_df,
            args.epochs, args.batch_size, args.device, output_dir
        )
        all_results[name] = results
    
    # Summary report
    summary_file = output_dir / "comparison_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY: ResNet Variants Comparison")
    print(f"{'='*80}")
    print(f"{'Variant':<12} {'Params':>10} {'Acc':>8} {'F1':>8} {'AIC':>10} {'BIC':>10} {'Infer(ms)':>10}")
    print("-" * 80)
    for name, res in all_results.items():
        print(f"{name:<12} {res['n_params']:>10,} {res['test_accuracy']:>8.3f} "
              f"{res['test_macro_f1']:>8.3f} {res['aic']:>10.1f} {res['bic']:>10.1f} "
              f"{res['inference_ms_per_sample']:>10.4f}")
    print(f"{'='*80}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
