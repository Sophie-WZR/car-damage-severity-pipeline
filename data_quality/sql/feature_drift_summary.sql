-- Lightweight split-level drift diagnostics for image metadata features.
WITH split_stats AS (
  SELECT
    split,
    AVG(width) AS avg_width,
    AVG(height) AS avg_height,
    AVG(aspect_ratio) AS avg_aspect_ratio,
    AVG(file_size_bytes) AS avg_file_size_bytes
  FROM image_manifest
  WHERE is_readable = 1
  GROUP BY split
),
train_stats AS (
  SELECT * FROM split_stats WHERE split = 'train'
),
heldout_stats AS (
  SELECT * FROM split_stats WHERE split = 'heldout_test'
)
SELECT
  'avg_width' AS feature,
  train_stats.avg_width AS train_mean,
  heldout_stats.avg_width AS heldout_mean,
  ABS(heldout_stats.avg_width - train_stats.avg_width) / NULLIF(train_stats.avg_width, 0) AS relative_gap
FROM train_stats, heldout_stats
UNION ALL
SELECT
  'avg_height',
  train_stats.avg_height,
  heldout_stats.avg_height,
  ABS(heldout_stats.avg_height - train_stats.avg_height) / NULLIF(train_stats.avg_height, 0)
FROM train_stats, heldout_stats
UNION ALL
SELECT
  'avg_aspect_ratio',
  train_stats.avg_aspect_ratio,
  heldout_stats.avg_aspect_ratio,
  ABS(heldout_stats.avg_aspect_ratio - train_stats.avg_aspect_ratio) / NULLIF(train_stats.avg_aspect_ratio, 0)
FROM train_stats, heldout_stats
UNION ALL
SELECT
  'avg_file_size_bytes',
  train_stats.avg_file_size_bytes,
  heldout_stats.avg_file_size_bytes,
  ABS(heldout_stats.avg_file_size_bytes - train_stats.avg_file_size_bytes) / NULLIF(train_stats.avg_file_size_bytes, 0)
FROM train_stats, heldout_stats
ORDER BY relative_gap DESC;
