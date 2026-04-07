# Data Quality Report

Generated at: 2026-04-07T17:08:59+00:00

## Dataset Overview

| split        | label    |   image_count |
|:-------------|:---------|--------------:|
| heldout_test | minor    |            82 |
| heldout_test | moderate |            75 |
| heldout_test | severe   |            91 |
| train        | minor    |           452 |
| train        | moderate |           463 |
| train        | severe   |           468 |

## Quality Checks

| metric_name                                        |   metric_value | scope                 | severity   | details                                                         |
|:---------------------------------------------------|---------------:|:----------------------|:-----------|:----------------------------------------------------------------|
| total_images                                       |      1631      | dataset               | info       |                                                                 |
| unreadable_images                                  |         0      | dataset               | pass       |                                                                 |
| images_in_exact_duplicate_groups                   |        22      | dataset               | warn       |                                                                 |
| exact_duplicate_hashes_across_splits               |         2      | dataset               | fail       | Same file hash appears in more than one split.                  |
| images_in_perceptual_duplicate_groups              |        38      | dataset               | warn       | Average-hash duplicate candidate count; review before deleting. |
| images_by_split                                    |       248      | heldout_test          | info       |                                                                 |
| images_by_split_label                              |        82      | heldout_test:minor    | info       |                                                                 |
| images_by_split_label                              |        75      | heldout_test:moderate | info       |                                                                 |
| images_by_split_label                              |        91      | heldout_test:severe   | info       |                                                                 |
| images_by_split                                    |      1383      | train                 | info       |                                                                 |
| images_by_split_label                              |       452      | train:minor           | info       |                                                                 |
| images_by_split_label                              |       463      | train:moderate        | info       |                                                                 |
| images_by_split_label                              |       468      | train:severe          | info       |                                                                 |
| images_by_label                                    |       534      | minor                 | info       |                                                                 |
| images_by_label                                    |       538      | moderate              | info       |                                                                 |
| images_by_label                                    |       559      | severe                | info       |                                                                 |
| relative_mean_gap_width_train_vs_heldout           |         0.0417 | dataset               | pass       | train_mean=246.67; heldout_mean=236.39                          |
| relative_mean_gap_height_train_vs_heldout          |         0.0395 | dataset               | pass       | train_mean=178.15; heldout_mean=171.12                          |
| relative_mean_gap_aspect_ratio_train_vs_heldout    |         0.0046 | dataset               | pass       | train_mean=1.41; heldout_mean=1.40                              |
| relative_mean_gap_file_size_bytes_train_vs_heldout |         0.0675 | dataset               | pass       | train_mean=8791.19; heldout_mean=8197.59                        |

## Warnings And Failures

| metric_name                           |   metric_value | scope   | severity   | details                                                         |
|:--------------------------------------|---------------:|:--------|:-----------|:----------------------------------------------------------------|
| exact_duplicate_hashes_across_splits  |              2 | dataset | fail       | Same file hash appears in more than one split.                  |
| images_in_exact_duplicate_groups      |             22 | dataset | warn       |                                                                 |
| images_in_perceptual_duplicate_groups |             38 | dataset | warn       | Average-hash duplicate candidate count; review before deleting. |

## Split Feature Summary

| split        |   width |   height |   aspect_ratio |   file_size_bytes |
|:-------------|--------:|---------:|---------------:|------------------:|
| heldout_test |  236.39 |   171.12 |           1.4  |           8197.59 |
| train        |  246.67 |   178.15 |           1.41 |           8791.19 |

## Exact Split-Leakage Candidates

| split        | label   | file_name   | sha256                                                           |
|:-------------|:--------|:------------|:-----------------------------------------------------------------|
| heldout_test | minor   | 0027.jpeg   | 42729fb0fef7230f6c1f5727ff3c0e18c330cd5e64f3771d9037ad102d88603e |
| train        | minor   | 0003.JPEG   | 42729fb0fef7230f6c1f5727ff3c0e18c330cd5e64f3771d9037ad102d88603e |
| heldout_test | minor   | 0004.JPEG   | 6f3372296185251320a5a020b53e78855d77ec15bfd013aa43cc7858fa045eaa |
| train        | minor   | 0263.JPEG   | 6f3372296185251320a5a020b53e78855d77ec15bfd013aa43cc7858fa045eaa |

## Model Metrics Summary

| model_name           |   best_val_macro_f1 |   heldout_test_acc |   heldout_test_macro_f1 |   heldout_test_macro_auc_ovr |
|:---------------------|--------------------:|-------------------:|------------------------:|-----------------------------:|
| resnet18_augmented   |            0.666179 |           0.721774 |                0.719332 |                     0.875694 |
| resnet18_baseline    |            0.642697 |           0.689516 |                0.692852 |                     0.864061 |
| resnet18_regularized |            0.575748 |           0.685484 |                0.677221 |                     0.846524 |

## Outputs

- Manifest CSV: `data_quality/dataset_manifest.csv`
- Metrics CSV: `data_quality/data_quality_metrics.csv`
- Duplicate candidates CSV: `data_quality/duplicate_candidates.csv`
- Split leakage CSV: `data_quality/split_leakage_exact_duplicates.csv`
- Model metrics CSV: `data_quality/model_metrics.csv`
- SQLite database: `data_quality/image_dataset.sqlite`
- Report: `data_quality/reports/data_quality_report.md`

## Resume-Relevant Summary

Built a reproducible image data quality layer that converts folder-based data into a queryable
manifest, validates image readability, checks duplicate and split-leakage risk, profiles class
balance, and monitors split-level distribution gaps before model training.
