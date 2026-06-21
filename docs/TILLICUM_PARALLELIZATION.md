# Parallelization & Training Larger ResNets on Tillicum

This guide covers (1) how to parallelize training across the 8 H200 GPUs of a Tillicum
node and (2) how to train the larger ResNets (ResNet101, ResNet152). Read
[`TILLICUM_TRAINING.md`](./TILLICUM_TRAINING.md) first — it covers access, the conda env,
staging the dataset, and the offline weight cache, all of which apply here too.

---

## First principles: what is there to parallelize?

Two facts decide the right strategy:

1. **The models are small relative to the GPU.** ResNet152 is ~60 M parameters
   (~230 MB fp32); a single H200 has **141 GB**. Even with activations and the Adam
   state, one variant uses a tiny fraction of one GPU. **A single model always fits — so
   there is nothing to split across GPUs. Model/pipeline parallelism is never needed
   here.**
2. **The dataset is small.** ~1,381 training images. A full epoch is seconds of GPU
   compute.

So "parallelize" here does **not** mean splitting a model. It means one of two things:

| Strategy | What it parallelizes | Best when | This repo |
|---|---|---|---|
| **A. Task parallel** | Different *models* run on different GPUs at once | You want to compare several variants | **Recommended** — run all 5 variants simultaneously |
| **B. Data parallel (DDP)** | One *model's* batch is split across GPUs | A single model's epoch is the bottleneck (large dataset/augmentation) | Marginal today; ready for when data grows |
| ~~C. Model parallel~~ | A model too big for one GPU | A single model doesn't fit in GPU memory | **Not applicable** — ResNet always fits |

> **Honest sizing note.** With 1,381 images, splitting one model's batch across GPUs
> (Strategy B) yields little real speedup — per-step gradient all-reduce and launch
> overhead rival the few-second epoch, and the effective batch can grow large enough to
> need LR retuning. The genuinely useful parallelism here is **Strategy A**: train every
> variant *at the same time*, each on its own GPU, so the whole comparison finishes in
> the time of one model. Reach for DDP when the dataset (or image size, or augmentation)
> grows enough that one model's epoch dominates.

---

## Strategy A — Task parallelism (recommended): one variant per GPU

A SLURM **job array** launches one independent single-GPU job per variant. They run
concurrently (subject to free GPUs), so the sweep's wall-clock is the slowest *single*
model rather than the sum of all of them.

Script: [`scripts/slurm/compare_resnet_array_tillicum.slurm`](../scripts/slurm/compare_resnet_array_tillicum.slurm)

```bash
cd $ALLOC/car-damage-severity-pipeline

# All five variants (array indices 0..4 -> resnet18/34/50/101/152):
sbatch scripts/slurm/compare_resnet_array_tillicum.slurm

# Or just the three larger models:
sbatch --array=2-4 scripts/slurm/compare_resnet_array_tillicum.slurm

squeue -u $USER          # you'll see one row per array task: <jobid>_0, _1, ...
```

Each task writes `<variant>_results.json` to a shared output dir (distinct filenames, so
no collisions). After the array finishes, aggregate into one report:

```bash
python tools/analyze_resnet_comparison.py \
  --results-dir /gpfs/scrubbed/$USER/car-damage/artifacts/compare \
  --output-report /gpfs/scrubbed/$USER/car-damage/artifacts/compare/analysis_report.md
```

> `analyze_resnet_comparison.py` globs every `resnet*_results.json` in the directory, so
> it correctly aggregates results no matter which array indices you ran or in what order
> they finished. (The `comparison_summary.json` from a single-variant run is **not** the
> aggregate — always aggregate with `analyze_*`.)

**Why this is the right default here:** five models, five GPUs, ~one model's worth of
wall-clock, and a clean apples-to-apples comparison on the fixed held-out test set.

---

## Strategy B — Data parallelism (DDP): scale one large model

When a *single* model's training is the bottleneck, use PyTorch
**DistributedDataParallel**: one process per GPU, each trains on a shard of the data,
gradients are all-reduced every step. Use it on **one node** (up to 8 GPUs); multi-node
adds NCCL/network complexity that this workload never warrants.

Script: [`tools/train_resnet_ddp.py`](../tools/train_resnet_ddp.py) launched via
`torchrun`, wrapped by
[`scripts/slurm/train_resnet_ddp_tillicum.slurm`](../scripts/slurm/train_resnet_ddp_tillicum.slurm).

```bash
cd $ALLOC/car-damage-severity-pipeline

# Default: resnet152 on 4 GPUs
sbatch scripts/slurm/train_resnet_ddp_tillicum.slurm

# resnet101 on all 8 GPUs of the node
sbatch --gpus=8 --export=ALL,VARIANT=resnet101 scripts/slurm/train_resnet_ddp_tillicum.slurm
```

How the script works:

- `torchrun --standalone --nproc_per_node=$NGPU` spawns one process per GPU and sets
  `RANK` / `WORLD_SIZE` / `LOCAL_RANK`. `train_resnet_ddp.py` auto-detects these and falls
  back to a plain single-process run when they're absent (so the same script works on
  your laptop with `python tools/train_resnet_ddp.py ...`).
- The training set is sharded with `DistributedSampler` (`set_epoch` each epoch);
  gradients all-reduce automatically through the DDP wrapper.
- **Evaluation runs on rank 0 only**, over the full (un-sharded) val/test sets, so the
  reported accuracy/F1 are exact (not a per-shard approximation).
- Only **rank 0** writes outputs: `<variant>_results.json` (schema-compatible with the
  comparison script, so `analyze_*` can read it) and `<variant>_weights.pt`.

### The one thing to get right: batch size and LR

`--batch-size` is **per GPU**. The effective global batch is
`batch_size × world_size`. With 8 GPUs at `--batch-size 64`, the global batch is 512.
Large global batches usually need the learning rate scaled up (≈ linearly) and/or a few
warmup epochs, or convergence degrades:

```bash
# 8 GPUs, keep global batch modest, scale LR with world size:
sbatch --gpus=8 --export=ALL,VARIANT=resnet152 \
  scripts/slurm/train_resnet_ddp_tillicum.slurm   # edit --batch-size / --lr in the script as needed
```

The results JSON records `world_size`, `per_gpu_batch_size`, and `global_batch_size` so
runs stay comparable.

---

## Training the larger ResNets (101 / 152)

ResNet101 and ResNet152 are registered in `tools/compare_resnet_variants.py`:

```python
ARCH = {
    "resnet18":  "resnet18.a1_in1k",
    "resnet34":  "resnet34.a1_in1k",
    "resnet50":  "resnet50.a1_in1k",
    "resnet101": "resnet101",   # bare name -> timm default pretrained tag
    "resnet152": "resnet152",
}
```

Parameter counts (trainable): resnet18 ≈ 11.2 M · resnet34 ≈ 21.8 M · resnet50 ≈ 25.6 M ·
**resnet101 ≈ 44.5 M · resnet152 ≈ 60.2 M**. All fit comfortably on one H200.

You can train a larger model three ways:

```bash
# 1) Single GPU via the comparison script (the default set stays 18/34/50):
python tools/compare_resnet_variants.py ... --variants resnet101
python tools/compare_resnet_variants.py ... --variants resnet101 resnet152   # both

# 2) Task-parallel (one GPU each) — see Strategy A:
sbatch --array=3-4 scripts/slurm/compare_resnet_array_tillicum.slurm

# 3) Data-parallel (multi-GPU, one model) — see Strategy B:
sbatch --export=ALL,VARIANT=resnet101 scripts/slurm/train_resnet_ddp_tillicum.slurm
```

### Cache the larger weights first (compute nodes may be offline)

ResNet101/152 pretrained weights must be in the `/gpfs` HF cache before the job runs,
because compute nodes may have no internet. Run on the **login node** (it has internet):

```bash
module load conda && conda activate $ALLOC/env
export HF_HOME=$ALLOC/hf_cache
python tools/prefetch_weights.py --variants resnet101 resnet152
# (or no args to cache every registered variant)
```

`prefetch_weights.py` reads the same `ARCH` table the trainer uses, so the cached tag can
never drift from what training requests.

### Verify a larger model actually trains (quick GPU smoke)

Before committing to a full run, confirm the model builds, fits, and steps on a real GPU:

```bash
salloc --account=macsvlarobotics --qos=interactive --gpus=2 --time=00:30:00
module load conda && conda activate $ALLOC/env
export HF_HOME=$ALLOC/hf_cache HF_HUB_OFFLINE=1
cd $ALLOC/car-damage-severity-pipeline
ln -sfn $SCRUBBED/data/training training && ln -sfn $SCRUBBED/data/validation validation

# (a) single-GPU sanity for resnet152:
python tools/train_resnet_ddp.py --variant resnet152 \
  --train-csv data_quality/clean_train_manifest.csv \
  --test-csv  data_quality/heldout_test_manifest.csv \
  --output-dir $SCRUBBED/artifacts/smoke --epochs 1 --batch-size 64

# (b) 2-GPU DDP sanity (exercises the all-reduce path):
torchrun --standalone --nproc_per_node=2 tools/train_resnet_ddp.py --variant resnet101 \
  --train-csv data_quality/clean_train_manifest.csv \
  --test-csv  data_quality/heldout_test_manifest.csv \
  --output-dir $SCRUBBED/artifacts/smoke --epochs 1 --batch-size 64
exit
```

If both write a `*_results.json` with sensible accuracy, the larger model + DDP path are
good to submit.

---

## Picking a strategy (decision guide)

```
Comparing several variants?  ─────► Strategy A (job array). One GPU each, runs in parallel.
Single model, small dataset? ─────► Single GPU (compare_resnet_variants.py --variants X).
Single model, big dataset /  ─────► Strategy B (DDP). torchrun on one node; scale LR with
  heavy augmentation, epoch                 global batch.
  is the bottleneck?
Model doesn't fit on a GPU?  ─────► N/A for ResNet. (Would be FSDP/pipeline — not needed.)
```

---

## Troubleshooting (parallel-specific)

| Symptom | Cause / fix |
|---|---|
| DDP job hangs at start | NCCL rendezvous. The script uses `--standalone` (single node); ensure `--nodes=1` and that `$NGPU` matches the granted GPUs (it's derived from `nvidia-smi -L`). |
| `RuntimeError: ... NCCL` / timeout mid-run | Uneven shards. The script uses `DistributedSampler` (equal-length shards) — don't bypass it. Keep all ranks on the same code path. |
| Array tasks all PENDING | Not enough free GPUs for all indices at once; they'll start as GPUs free. Narrow with `--array=` or lower concurrency with `--array=0-4%2`. |
| Worse accuracy with many GPUs | Global batch too large for the base LR. Scale `--lr` ~linearly with `world_size`, or reduce `--batch-size`. |
| `No training images resolved` | Dataset not staged / symlinked. See `TILLICUM_TRAINING.md` §4 — run from the repo root with `training/` and `validation/` linked in. |
| Weights download hangs | Compute node offline. Prefetch on the login node (`prefetch_weights.py`) and keep `HF_HUB_OFFLINE=1`. |

---

## Reference: files added for parallel / larger-model training

| File | Purpose |
|---|---|
| `tools/train_resnet_ddp.py` | Single-variant trainer; single-GPU or multi-GPU DDP (auto-detected) |
| `tools/prefetch_weights.py` | Cache timm pretrained weights into `HF_HOME` (login node) |
| `scripts/slurm/train_resnet_ddp_tillicum.slurm` | DDP job (Strategy B), one node, N GPUs |
| `scripts/slurm/compare_resnet_array_tillicum.slurm` | Job array (Strategy A), one variant per GPU |
| `tools/compare_resnet_variants.py` | `--variants` now includes `resnet101` / `resnet152` |
