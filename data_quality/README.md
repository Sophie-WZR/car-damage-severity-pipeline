# Data Quality Layer

This folder adds a Data Engineering layer to the car damage severity project.
It converts the folder-based image dataset into a queryable manifest, validates
image quality, checks duplicate and split-leakage risk, and writes summary
outputs for SQL analysis.

## Run

```bash
python data_quality/build_dataset_manifest.py
python data_quality/create_clean_split.py
```

## Outputs

- `dataset_manifest.csv`: one row per image with label, split, file metadata,
  dimensions, SHA-256 hash, perceptual hash, and readability status.
- `data_quality_metrics.csv`: data quality and drift-style summary metrics.
- `duplicate_candidates.csv`: exact and perceptual duplicate candidates for review.
- `split_leakage_exact_duplicates.csv`: exact duplicate rows that appear across splits.
- `model_metrics.csv`: model metric JSON files normalized into a tabular summary.
- `clean_train_manifest.csv`: training rows after removing exact duplicate hashes
  that appear in the held-out test split.
- `heldout_test_manifest.csv`: fixed held-out test rows.
- `removed_train_leakage_rows.csv`: excluded training rows for leakage auditing.
- `image_dataset.sqlite`: queryable SQLite database with `image_manifest` and
  `data_quality_metrics`, `duplicate_candidates`,
  `split_leakage_exact_duplicates`, `clean_train_manifest`,
  `heldout_test_manifest`, `removed_train_leakage_rows`, and `model_metrics`
  tables.
- `reports/data_quality_report.md`: human-readable report for portfolio use.

## SQL Examples

```bash
sqlite3 data_quality/image_dataset.sqlite < data_quality/sql/class_distribution.sql
sqlite3 data_quality/image_dataset.sqlite < data_quality/sql/split_quality_checks.sql
sqlite3 data_quality/image_dataset.sqlite < data_quality/sql/feature_drift_summary.sql
sqlite3 data_quality/image_dataset.sqlite < data_quality/sql/model_metrics_summary.sql
```
