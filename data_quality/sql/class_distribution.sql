-- Class distribution by split, with within-split percentages.
SELECT
  split,
  label,
  COUNT(*) AS image_count,
  ROUND(
    100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY split),
    2
  ) AS pct_within_split
FROM image_manifest
GROUP BY split, label
ORDER BY split, label;
