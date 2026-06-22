# Weights & Biases (wandb) Integration

This guide covers how to set up and use Weights & Biases to log training metrics for the ResNet152 model.

## Quick Start

### 1. Install wandb and login

```bash
pip install wandb
wandb login
# Follow the prompts to enter your API key from https://wandb.ai/
```

### 2. Run training with wandb logging

**Local training:**
```bash
python tools/train_resnet_ddp.py \
  --variant resnet152 \
  --train-csv data_quality/clean_train_manifest.csv \
  --test-csv data_quality/heldout_test_manifest.csv \
  --output-dir artifacts/resnet152 \
  --epochs 12 \
  --batch-size 64 \
  --wandb-project "car-damage-severity" \
  --wandb-entity "your-username"
```

**On Tillicum:**
```bash
sbatch scripts/slurm/train_resnet152_tillicum.slurm
```

The SLURM script logs to `car-damage-severity` project by default.

### 3. View results

Open https://wandb.ai to see real-time training curves and metrics.

---

## Logged Metrics

### Per-Epoch Metrics
- **train_loss**: Cross-entropy loss on training set
- **val_accuracy**: Accuracy on validation set
- **val_f1**: Macro-F1 score on validation set

### Final Test Metrics
- **test_accuracy**: Accuracy on held-out test set
- **test_f1**: Macro-F1 score on held-out test set
- **n_params**: Number of trainable parameters
- **wall_time_sec**: Total training wall-clock time
- **aic**: Akaike Information Criterion
- **bic**: Bayesian Information Criterion

### Config (logged once per run)
- **variant**: Model architecture (e.g., "resnet152")
- **epochs**: Number of training epochs
- **batch_size_per_gpu**: Batch size per GPU
- **global_batch_size**: Effective batch size across all GPUs
- **learning_rate**: Initial learning rate
- **world_size**: Number of GPUs in DDP setup
- **num_workers**: DataLoader worker threads per process

---

## Command-Line Options

When running `tools/train_resnet_ddp.py`:

```bash
--wandb-project PROJECT   # wandb project name (default: "car-damage-severity")
--wandb-entity ENTITY     # wandb entity/username (default: None = personal workspace)
--disable-wandb           # Disable wandb logging (useful for testing)
```

---

## Comparing Runs

Once multiple runs are logged, you can:
1. Go to your wandb project page
2. Use the **Report** feature to create custom comparisons
3. Export CSV data for further analysis

Example comparison:
- Filter by `variant=resnet152`, `world_size=1`
- Compare test_accuracy across different batch sizes or learning rates

---

## Offline Mode (Compute Nodes)

If the compute node has no internet access, wandb gracefully degrades:
- Metrics are cached locally
- After the job finishes, re-sync to wandb:
  ```bash
  wandb sync .
  ```

This is handled automatically in the SLURM script.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Not logged in" error | Run `wandb login` on login node with internet access |
| Metrics not appearing | Ensure `--wandb-project` and `--wandb-entity` match your account |
| Run name is cryptic | Runs auto-name with `variant-timestamp` (visible in wandb UI) |
| Out of API calls | Check wandb plan limits at https://wandb.ai/billing |

---

## References

- [Weights & Biases Documentation](https://docs.wandb.ai)
- [wandb Python API](https://docs.wandb.ai/ref/python)
- [Logging Hyperparameters](https://docs.wandb.ai/guides/runs/config)
