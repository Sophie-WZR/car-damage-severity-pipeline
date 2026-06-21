# ResNet Variant Comparison Guide

This document explains how to compare ResNet18/34/50 architectures on the
car damage severity classification task.

## Background

Current baseline model: ResNet18 (11.2M parameters)
Models to compare:
- ResNet34: 21.8M parameters
- ResNet50: 25.6M parameters

## Quick Start

### 1. Verify Dataset

```bash
ls data_quality/clean_train_manifest.csv
ls data_quality/heldout_test_manifest.csv
```

### 2. Run Comparison

```bash
python tools/compare_resnet_variants.py \
  --train-csv data_quality/clean_train_manifest.csv \
  --test-csv data_quality/heldout_test_manifest.csv \
  --output-dir artifacts/resnet_comparison \
  --epochs 8 \
  --batch-size 32
```

### 3. Check Results

Results are saved to `artifacts/resnet_comparison/`:
- `resnet18_results.json`: detailed ResNet18 results
- `resnet34_results.json`: detailed ResNet34 results
- `resnet50_results.json`: detailed ResNet50 results
- `comparison_summary.json`: aggregated comparison for analysis

## Output Metrics

Each model evaluation includes:

| Metric | Description |
|--------|-------------|
| **n_params** | Number of trainable parameters (millions) |
| **test_accuracy** | Accuracy on held-out test set |
| **test_macro_f1** | Macro-averaged F1 score (handles class imbalance) |
| **test_report** | Per-class precision/recall/F1 scores |
| **confusion_matrix** | Prediction vs. ground truth matrix |
| **aic** | Akaike Information Criterion (lower = better balance of fit and complexity) |
| **bic** | Bayesian Information Criterion (penalizes larger models more heavily) |
| **inference_ms_per_sample** | Average inference time in milliseconds |

## Model Selection Criteria

1. **Accuracy-first approach**: Select the model with highest `test_accuracy` or `test_macro_f1`.
2. **Complexity-aware**: Select the model with lowest `AIC` or `BIC` to avoid overfitting.
3. **Speed-aware**: Among candidates with similar accuracy, prefer faster inference.

**Recommendation**: If ResNet50 achieves significantly lower AIC/BIC with higher
accuracy than ResNet18, upgrade to ResNet50. Otherwise, retain ResNet18 for
computational efficiency.

## Post-Analysis

After comparison completes, you can:

1. Generate a markdown analysis report:
   ```bash
   python tools/analyze_resnet_comparison.py \
     --results-dir artifacts/resnet_comparison \
     --output-report artifacts/resnet_comparison/analysis_report.md
   ```

2. Plot model accuracy vs. parameters to visualize the efficiency frontier.
3. Inspect training curves to detect overfitting (train_loss vs. val_loss).
4. Run 5-fold cross-validation on the best model for robust performance estimates.

## Troubleshooting

- **Out of memory**: Lower `--batch-size` (e.g., 16) or use smaller architectures.
- **Slow training**: Ensure GPU is available (auto-detected) or reduce `--epochs`.
- **Manifest path errors**: Verify CSV files contain `file_path` and `label` columns;
  the script auto-resolves paths by matching filenames to local `training/`
  and `validation/` directories using the `source_label_dir` hint.

## File Structure

```
car-damage-severity-pipeline/
├── tools/
│   ├── compare_resnet_variants.py      # main comparison script
│   └── analyze_resnet_comparison.py    # result analysis and reporting
├── data_quality/
│   ├── clean_train_manifest.csv        # training split manifest
│   └── heldout_test_manifest.csv       # held-out test manifest
├── artifacts/
│   └── resnet_comparison/              # output directory
│       ├── resnet18_results.json
│       ├── resnet34_results.json
│       ├── resnet50_results.json
│       ├── comparison_summary.json     # for aggregated analysis
│       └── analysis_report.md          # generated markdown report
└── requirements.txt                    # dependencies (torch, timm, etc.)
```

