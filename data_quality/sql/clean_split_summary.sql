-- Summary of leakage-aware clean train and fixed held-out test manifests.
SELECT metric, value
FROM clean_split_summary
ORDER BY metric;

-- Class counts after removing exact train/test duplicate leakage from train.
SELECT
  'clean_train' AS split,
  label,
  COUNT(*) AS image_count
FROM clean_train_manifest
GROUP BY label
UNION ALL
SELECT
  'heldout_test' AS split,
  label,
  COUNT(*) AS image_count
FROM heldout_test_manifest
GROUP BY label
ORDER BY split, label;
