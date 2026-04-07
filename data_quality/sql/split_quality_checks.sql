-- Duplicate and split-leakage checks for the image dataset.

-- Exact duplicate files by SHA-256 hash.
SELECT
  sha256,
  COUNT(*) AS image_count,
  GROUP_CONCAT(DISTINCT split) AS splits,
  GROUP_CONCAT(DISTINCT label) AS labels
FROM image_manifest
GROUP BY sha256
HAVING COUNT(*) > 1
ORDER BY image_count DESC, sha256;

-- Exact same image bytes appearing in multiple splits.
SELECT
  sha256,
  COUNT(*) AS image_count,
  COUNT(DISTINCT split) AS split_count,
  GROUP_CONCAT(DISTINCT split) AS splits,
  GROUP_CONCAT(file_path, ' | ') AS file_paths
FROM image_manifest
GROUP BY sha256
HAVING COUNT(DISTINCT split) > 1
ORDER BY image_count DESC, sha256;

-- Unreadable image files.
SELECT
  split,
  label,
  file_path,
  error
FROM image_manifest
WHERE is_readable = 0
ORDER BY split, label, file_path;
