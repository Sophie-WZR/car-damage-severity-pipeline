# Car Damage Severity Data Pipeline

This project combines a leakage-aware image data quality pipeline with a PyTorch
classification model for car damage severity prediction.

## Dataset

The raw image dataset was sourced from Kaggle and is not redistributed in this
repository. To reproduce the pipeline, download the Kaggle dataset according to
its original license and place the image folders at:

- `training/`
- `validation/`

Then run the data quality scripts below to regenerate the manifests, clean
split files, and SQLite quality tables.

## Data Engineering Layer

The `data_quality/` workflow converts the folder-based image dataset into a
queryable manifest and SQLite database, then runs data quality checks before
model training.

```bash
python data_quality/build_dataset_manifest.py
python data_quality/create_clean_split.py
```

Key outputs:

- `portfolio_site/index.html`: static results page for recruiter and portfolio review.
- `portfolio_site/quality-report.html`: browser-friendly data quality report.
- `data_quality/dataset_manifest.csv`: one row per image with split, label,
  path, image dimensions, file metadata, SHA-256 hash, perceptual hash, and
  readability status.
- `data_quality/image_dataset.sqlite`: queryable tables for image metadata,
  quality metrics, duplicate candidates, split-leakage rows, clean train rows,
  held-out test rows, and model metrics.
- `data_quality/clean_train_manifest.csv`: training manifest after removing
  exact duplicate hashes that appear in held-out test.
- `data_quality/reports/data_quality_report.md`: data quality summary for
  portfolio review.

The quality layer detected 2 exact duplicate hashes across train and held-out
test and removed the 2 corresponding training rows from the clean training
manifest while keeping the held-out test set fixed.

## Model Layer

`Model3-CNN.ipynb` uses the leakage-aware clean manifests when present. It
splits `clean_train_manifest.csv` into internal train/validation data for model
selection, then evaluates once on the fixed held-out test manifest.

### ResNet Variant Comparison

New tool `tools/compare_resnet_variants.py` enables automated comparison of
ResNet18, ResNet34, ResNet50, and ResNet101 architectures on the same dataset.
Features:
- Automatic path resolution for manifest CSV files pointing to external directories
- Stratified train/validation/test splits with preserved class balance
- Per-model evaluation: accuracy, macro-F1, per-class precision/recall/F1, AIC/BIC
- Inference time benchmarking

Usage:
```bash
python tools/compare_resnet_variants.py \
  --train-csv data_quality/clean_train_manifest.csv \
  --test-csv data_quality/heldout_test_manifest.csv \
  --output-dir artifacts/resnet_comparison \
  --epochs 8 \
  --batch-size 32
```

### Latest Results (ResNet18)

- Model: ImageNet-pretrained ResNet18
- Held-out test accuracy: **87.9%** (updated with path fix)
- Held-out test macro-F1: **87.6%**
- Parameters: 11.2M
- Training manifests: `data_quality/clean_train_manifest.csv`
- Detailed results: `artifacts/resnet_comparison/resnet18_testfix_results.json`

The final model artifacts were generated from the leakage-aware clean training
split and fixed held-out test manifest.

## Portfolio Page

Open `portfolio_site/index.html` directly in a browser, or preview it locally:

```bash
python -m http.server 8000
```

Then visit `http://localhost:8000/portfolio_site/`.

## SQL Examples

```bash
sqlite3 data_quality/image_dataset.sqlite < data_quality/sql/class_distribution.sql
sqlite3 data_quality/image_dataset.sqlite < data_quality/sql/split_quality_checks.sql
sqlite3 data_quality/image_dataset.sqlite < data_quality/sql/clean_split_summary.sql
sqlite3 data_quality/image_dataset.sqlite < data_quality/sql/model_metrics_summary.sql
```

## Resume Summary

Built a reproducible image data quality and model evaluation pipeline with
Python and SQL, including dataset manifest generation, duplicate detection,
split-leakage remediation, class distribution monitoring, and held-out model
evaluation for a 3-class car damage severity classifier.
