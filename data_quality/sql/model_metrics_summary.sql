-- Model metric JSON outputs normalized into a queryable table.
SELECT
  model_name,
  selection_metric,
  ROUND(best_val_macro_f1, 4) AS best_val_macro_f1,
  ROUND(heldout_test_acc, 4) AS heldout_test_acc,
  ROUND(heldout_test_macro_f1, 4) AS heldout_test_macro_f1,
  ROUND(heldout_test_macro_auc_ovr, 4) AS heldout_test_macro_auc_ovr
FROM model_metrics
ORDER BY heldout_test_macro_f1 DESC;
