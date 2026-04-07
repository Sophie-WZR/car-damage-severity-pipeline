#!/usr/bin/env python3
"""Build a queryable image dataset manifest and data quality report.

This script is intentionally independent from the model-training notebook. It
turns the folder-based image dataset into CSV + SQLite tables, then runs data
quality checks that are useful for a Data Engineering portfolio story:
readability, class balance, duplicate detection, split leakage, and feature
drift-style checks across dataset splits.
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASSES = ("minor", "moderate", "severe")


@dataclass(frozen=True)
class QualityConfig:
    training_dir: Path
    heldout_dir: Path
    output_dir: Path

    @property
    def manifest_csv(self) -> Path:
        return self.output_dir / "dataset_manifest.csv"

    @property
    def metrics_csv(self) -> Path:
        return self.output_dir / "data_quality_metrics.csv"

    @property
    def duplicate_candidates_csv(self) -> Path:
        return self.output_dir / "duplicate_candidates.csv"

    @property
    def split_leakage_csv(self) -> Path:
        return self.output_dir / "split_leakage_exact_duplicates.csv"

    @property
    def model_metrics_csv(self) -> Path:
        return self.output_dir / "model_metrics.csv"

    @property
    def db_path(self) -> Path:
        return self.output_dir / "image_dataset.sqlite"

    @property
    def report_path(self) -> Path:
        return self.output_dir / "reports" / "data_quality_report.md"


def label_from_dir(name: str) -> str | None:
    normalized = name.lower()
    for label in CLASSES:
        if label in normalized:
            return label
    return None


def scan_files(root: Path, split: str) -> Iterable[dict]:
    for label_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        label = label_from_dir(label_dir.name)
        if label is None:
            continue
        for file_path in sorted(label_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in IMG_EXT:
                yield {
                    "split": split,
                    "label": label,
                    "source_label_dir": label_dir.name,
                    "file_path": str(file_path.resolve()),
                    "file_name": file_path.name,
                    "file_ext": file_path.suffix.lower(),
                }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_hash(image: Image.Image, hash_size: int = 8) -> str:
    """Compute a simple perceptual hash without requiring extra packages."""
    gray = image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float32)
    bits = pixels > pixels.mean()
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:0{hash_size * hash_size // 4}x}"


def image_metadata(path: Path) -> dict:
    base = {
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "is_readable": False,
        "width": None,
        "height": None,
        "aspect_ratio": None,
        "mode": None,
        "perceptual_hash": None,
        "error": None,
    }
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            base.update(
                {
                    "is_readable": True,
                    "width": int(width),
                    "height": int(height),
                    "aspect_ratio": float(width / height) if height else None,
                    "mode": image.mode,
                    "perceptual_hash": average_hash(image),
                }
            )
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"
    return base


def build_manifest(config: QualityConfig) -> pd.DataFrame:
    records = []
    for raw in list(scan_files(config.training_dir, "train")) + list(
        scan_files(config.heldout_dir, "heldout_test")
    ):
        path = Path(raw["file_path"])
        raw.update(image_metadata(path))
        records.append(raw)

    manifest = pd.DataFrame(records)
    if manifest.empty:
        raise RuntimeError("No images found. Check the training and heldout directories.")

    manifest = manifest.sort_values(["split", "label", "file_name"]).reset_index(drop=True)
    manifest.insert(
        0,
        "image_id",
        [
            hashlib.sha1(f"{row.split}|{row.label}|{row.file_path}".encode("utf-8")).hexdigest()[:16]
            for row in manifest.itertuples()
        ],
    )
    manifest["ingested_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return manifest


def build_quality_metrics(manifest: pd.DataFrame) -> pd.DataFrame:
    metrics: list[dict] = []

    def add(name: str, value, scope: str = "dataset", severity: str = "info", details: str = "") -> None:
        metrics.append(
            {
                "metric_name": name,
                "metric_value": value,
                "scope": scope,
                "severity": severity,
                "details": details,
            }
        )

    total = len(manifest)
    unreadable = int((~manifest["is_readable"]).sum())
    add("total_images", total)
    add("unreadable_images", unreadable, severity="fail" if unreadable else "pass")

    exact_dup_groups = manifest.groupby("sha256")["image_id"].transform("size")
    exact_dups = int((exact_dup_groups > 1).sum())
    add("images_in_exact_duplicate_groups", exact_dups, severity="warn" if exact_dups else "pass")

    split_leakage = (
        manifest.groupby("sha256")["split"].nunique().reset_index(name="split_count").query("split_count > 1")
    )
    add(
        "exact_duplicate_hashes_across_splits",
        int(len(split_leakage)),
        severity="fail" if len(split_leakage) else "pass",
        details="Same file hash appears in more than one split.",
    )

    perceptual_dup_groups = manifest.dropna(subset=["perceptual_hash"]).groupby("perceptual_hash")[
        "image_id"
    ].transform("size")
    perceptual_dups = int((perceptual_dup_groups > 1).sum())
    add(
        "images_in_perceptual_duplicate_groups",
        perceptual_dups,
        severity="warn" if perceptual_dups else "pass",
        details="Average-hash duplicate candidate count; review before deleting.",
    )

    readable = manifest[manifest["is_readable"]].copy()
    for split, group in manifest.groupby("split"):
        add("images_by_split", int(len(group)), scope=split)
        for label in CLASSES:
            add("images_by_split_label", int((group["label"] == label).sum()), scope=f"{split}:{label}")

    for label, count in manifest["label"].value_counts().reindex(CLASSES, fill_value=0).items():
        add("images_by_label", int(count), scope=label)

    label_dist = (
        manifest.groupby(["split", "label"]).size().groupby(level=0).apply(lambda s: s / s.sum()).unstack()
    )
    if {"train", "heldout_test"}.issubset(label_dist.index):
        max_label_gap = float((label_dist.loc["train"] - label_dist.loc["heldout_test"]).abs().max())
        add(
            "max_class_distribution_gap_train_vs_heldout",
            round(max_label_gap, 4),
            severity="warn" if max_label_gap > 0.10 else "pass",
            details="Maximum absolute class-share difference between training and held-out test.",
        )

    if not readable.empty and {"train", "heldout_test"}.issubset(set(readable["split"])):
        for column in ["width", "height", "aspect_ratio", "file_size_bytes"]:
            split_means = readable.groupby("split")[column].mean()
            train_mean = float(split_means.get("train", np.nan))
            test_mean = float(split_means.get("heldout_test", np.nan))
            if np.isfinite(train_mean) and np.isfinite(test_mean) and train_mean:
                rel_gap = abs(test_mean - train_mean) / train_mean
                add(
                    f"relative_mean_gap_{column}_train_vs_heldout",
                    round(float(rel_gap), 4),
                    severity="warn" if rel_gap > 0.25 else "pass",
                    details=f"train_mean={train_mean:.2f}; heldout_mean={test_mean:.2f}",
                )

    return pd.DataFrame(metrics)


def duplicate_candidates(manifest: pd.DataFrame) -> pd.DataFrame:
    exact_counts = manifest.groupby("sha256")["image_id"].transform("size")
    perceptual_counts = manifest.groupby("perceptual_hash")["image_id"].transform("size")
    dupes = manifest[(exact_counts > 1) | (perceptual_counts > 1)].copy()
    return dupes.sort_values(["sha256", "perceptual_hash", "split", "label", "file_name"])


def split_leakage_exact_duplicates(manifest: pd.DataFrame) -> pd.DataFrame:
    split_counts = manifest.groupby("sha256")["split"].transform("nunique")
    leaked = manifest[split_counts > 1].copy()
    return leaked.sort_values(["sha256", "split", "label", "file_name"])


def load_model_metrics(artifacts_dir: Path = Path("artifacts")) -> pd.DataFrame:
    """Collect model metric JSON files into a table for SQL analysis."""
    import json

    metric_files = {
        "resnet18_baseline": artifacts_dir / "metrics_resnet18_baseline_heldout.json",
        "resnet18_regularized": artifacts_dir / "metrics_resnet18_regularized_heldout.json",
        "resnet18_augmented": artifacts_dir / "metrics_resnet18_augmented_heldout.json",
    }
    rows = []
    for model_name, path in metric_files.items():
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows.append({"model_name": model_name, "metrics_path": str(path), **payload})
    return pd.DataFrame(rows)


def write_sqlite(
    manifest: pd.DataFrame,
    metrics: pd.DataFrame,
    duplicates: pd.DataFrame,
    leakage: pd.DataFrame,
    model_metrics: pd.DataFrame,
    db_path: Path,
) -> None:
    with sqlite3.connect(db_path) as conn:
        manifest.to_sql("image_manifest", conn, if_exists="replace", index=False)
        metrics.to_sql("data_quality_metrics", conn, if_exists="replace", index=False)
        duplicates.to_sql("duplicate_candidates", conn, if_exists="replace", index=False)
        leakage.to_sql("split_leakage_exact_duplicates", conn, if_exists="replace", index=False)
        model_metrics.to_sql("model_metrics", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_manifest_split_label ON image_manifest(split, label)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_manifest_sha256 ON image_manifest(sha256)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_manifest_perceptual_hash ON image_manifest(perceptual_hash)"
        )


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.to_markdown(index=False)


def write_report(
    manifest: pd.DataFrame,
    metrics: pd.DataFrame,
    duplicates: pd.DataFrame,
    leakage: pd.DataFrame,
    model_metrics: pd.DataFrame,
    config: QualityConfig,
) -> None:
    config.report_path.parent.mkdir(parents=True, exist_ok=True)

    counts = (
        manifest.groupby(["split", "label"])
        .size()
        .reset_index(name="image_count")
        .sort_values(["split", "label"])
    )
    failures = metrics[metrics["severity"].isin(["fail", "warn"])].sort_values(
        ["severity", "metric_name", "scope"]
    )
    readable = manifest[manifest["is_readable"]]
    feature_summary = (
        readable.groupby("split")[["width", "height", "aspect_ratio", "file_size_bytes"]]
        .mean()
        .round(2)
        .reset_index()
    )
    leakage_preview = leakage[["split", "label", "file_name", "sha256"]].head(20)
    model_columns = [
        col
        for col in [
            "model_name",
            "best_val_macro_f1",
            "heldout_test_acc",
            "heldout_test_macro_f1",
            "heldout_test_macro_auc_ovr",
        ]
        if col in model_metrics.columns
    ]
    model_summary = model_metrics[model_columns].sort_values(
        "heldout_test_macro_f1", ascending=False
    ) if model_columns and "heldout_test_macro_f1" in model_metrics.columns else model_metrics

    text = f"""# Data Quality Report

Generated at: {datetime.now(timezone.utc).isoformat(timespec="seconds")}

## Dataset Overview

{markdown_table(counts)}

## Quality Checks

{markdown_table(metrics)}

## Warnings And Failures

{markdown_table(failures)}

## Split Feature Summary

{markdown_table(feature_summary)}

## Exact Split-Leakage Candidates

{markdown_table(leakage_preview)}

## Model Metrics Summary

{markdown_table(model_summary)}

## Outputs

- Manifest CSV: `{config.manifest_csv}`
- Metrics CSV: `{config.metrics_csv}`
- Duplicate candidates CSV: `{config.duplicate_candidates_csv}`
- Split leakage CSV: `{config.split_leakage_csv}`
- Model metrics CSV: `{config.model_metrics_csv}`
- SQLite database: `{config.db_path}`
- Report: `{config.report_path}`

## Resume-Relevant Summary

Built a reproducible image data quality layer that converts folder-based data into a queryable
manifest, validates image readability, checks duplicate and split-leakage risk, profiles class
balance, and monitors split-level distribution gaps before model training.
"""
    config.report_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=Path("training"))
    parser.add_argument("--heldout-dir", type=Path, default=Path("validation"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_quality"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = QualityConfig(args.training_dir, args.heldout_dir, args.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(config)
    metrics = build_quality_metrics(manifest)
    duplicates = duplicate_candidates(manifest)
    leakage = split_leakage_exact_duplicates(manifest)
    model_metrics = load_model_metrics()

    manifest.to_csv(config.manifest_csv, index=False)
    metrics.to_csv(config.metrics_csv, index=False)
    duplicates.to_csv(config.duplicate_candidates_csv, index=False)
    leakage.to_csv(config.split_leakage_csv, index=False)
    model_metrics.to_csv(config.model_metrics_csv, index=False)
    write_sqlite(manifest, metrics, duplicates, leakage, model_metrics, config.db_path)
    write_report(manifest, metrics, duplicates, leakage, model_metrics, config)

    print(f"Wrote {len(manifest):,} manifest rows to {config.manifest_csv}")
    print(f"Wrote {len(metrics):,} quality metrics to {config.metrics_csv}")
    print(f"Wrote {len(duplicates):,} duplicate candidate rows to {config.duplicate_candidates_csv}")
    print(f"Wrote {len(leakage):,} split-leakage rows to {config.split_leakage_csv}")
    print(f"Wrote {len(model_metrics):,} model metric rows to {config.model_metrics_csv}")
    print(f"Wrote SQLite database to {config.db_path}")
    print(f"Wrote report to {config.report_path}")


if __name__ == "__main__":
    main()
