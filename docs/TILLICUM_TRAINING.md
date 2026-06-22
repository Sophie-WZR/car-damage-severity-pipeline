# Training the Bigger ResNet on Tillicum (UW Hyak)

This guide walks through submitting a SLURM job on UW Hyak **Tillicum** to train the
larger ResNet (ResNet50) for the car-damage-severity classifier. It adapts the
DreamZero `TILLICUM_SETUP.md` to this much smaller workload.

> **How this differs from the DreamZero run.** This is a *tiny* job — ~1,381 training
> images, ~248 held-out test images, a single GPU, a few minutes of compute. There is
> **no DeepSpeed, flash-attn, NCCL, or multi-GPU** here. You will use far less than the
> free 100 GPU-hr tier. The two things that actually bite on this cluster are (1) the
> manifest CSVs store *foreign absolute paths*, and (2) compute nodes may have **no
> internet**, so pretrained weights must be cached first. Both are handled below.

Tillicum at a glance: GPU-only SLURM cluster, NVIDIA **H200 (141 GB)**, 8 per node,
~8 CPU + ~200 GB RAM per GPU. No `sudo`/`apt`. Account **macsvlarobotics**
(worktag PG225985), 100 free GPU-hr then `$0.90/GPU-hr`.

Sequence: **access → storage → env → repo → stage data + cache weights → verify →
(smoke) → submit job → monitor → retrieve results.**

---

## 0. Access (no VPN needed for SSH)

```bash
ssh <UWNetID>@tillicum.hyak.uw.edu      # approve the Duo push / enter passcode
hostname                                 # confirms a login node, e.g. tillicum-login01
```

> Login nodes are for **setup, staging, and `sbatch` submission only** — never run
> training on them. Repeated failed logins → ~1-hour IP ban; make sure Duo 2FA is
> enrolled at identity.uw.edu first.

---

## 1. Storage layout

| Path | Quota / policy | Use for |
|---|---|---|
| `/gpfs/home/<netid>` | 10 GB, backed up | dotfiles only |
| `/gpfs/projects/macsvlarobotics` | 1 TB, backed up, **purged at project end** | repo, conda env, weight cache |
| `/gpfs/scrubbed/<netid>` | large, **no backup, purged after 60d idle** | dataset, logs, artifacts |

Keep this project in its **own subdirectory** so it doesn't collide with DreamZero:

```bash
ALLOC=/gpfs/projects/macsvlarobotics/car-damage
SCRUBBED=/gpfs/scrubbed/$USER/car-damage
mkdir -p $ALLOC $SCRUBBED/logs $SCRUBBED/data $SCRUBBED/artifacts
```

If you don't already have one, keep conda off the 10 GB home dir via `~/.condarc`:

```yaml
envs_dirs:
  - /gpfs/projects/macsvlarobotics/conda/envs
pkgs_dirs:
  - /gpfs/projects/macsvlarobotics/conda/pkgs
```

> Use whatever allocation you're a member of. If it isn't `macsvlarobotics`, replace
> the account name here, in `~/.condarc`, and in the `#SBATCH --account=` line of the
> job script.

---

## 2. Conda environment (a dedicated, minimal env)

This project needs only torch + a few light libraries — build a fresh env rather than
reusing the heavy DreamZero one.

```bash
module load conda
conda create --prefix $ALLOC/env python=3.11 -y
conda activate $ALLOC/env
```

Install PyTorch with a CUDA build that supports the H200 (Hopper, sm_90), then the rest
of the requirements:

```bash
# H200 needs a recent CUDA wheel; cu124 works. Match the closest available if needed.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install timm scikit-learn pandas numpy pillow matplotlib seaborn
```

> For multi-GPU parallelization and the larger ResNets (101/152), see
> [`TILLICUM_PARALLELIZATION.md`](./TILLICUM_PARALLELIZATION.md).

> torch ships its **own** CUDA/cuDNN libraries — do **not** `module load cuda` at
> runtime (it shadows torch's cuDNN). The job script deliberately loads only `conda`.

---

## 3. Clone the repo

```bash
cd $ALLOC
git clone <repo-url> car-damage-severity-pipeline
cd car-damage-severity-pipeline
```

Layout the job script assumes: repo at `$ALLOC/car-damage-severity-pipeline`, env at
`$ALLOC/env` (sibling), dataset on `$SCRUBBED/data`, weight cache at `$ALLOC/hf_cache`.

---

## 4. Stage the dataset (login node → `/gpfs/scrubbed`)

The Kaggle image folders are **not** in the repo. Copy your local `training/` and
`validation/` folders (the ones with `01-minor` / `02-moderate` / `03-severe`
subfolders) up to scrubbed. Run this **from your own machine**:

```bash
rsync -avz ./training/   <netid>@tillicum.hyak.uw.edu:/gpfs/scrubbed/<netid>/car-damage/data/training/
rsync -avz ./validation/ <netid>@tillicum.hyak.uw.edu:/gpfs/scrubbed/<netid>/car-damage/data/validation/
```

> **Why this matters.** The manifest CSVs (`clean_train_manifest.csv`,
> `heldout_test_manifest.csv`) store absolute paths from the machine that built them
> (`/Users/wangzhuoran/Desktop/...`), which don't exist on Tillicum. The training
> script falls back to finding each image by **filename** inside `./training` and
> `./validation` relative to the working directory. The job script `cd`s into the repo
> and symlinks the staged folders in, so that fallback resolves. Stage the folders with
> the same class-subfolder names and you're done — no CSV editing needed.

---

## 5. Cache pretrained weights (login node — it has internet)

`timm.create_model("resnet50.a1_in1k", pretrained=True)` downloads weights from
HuggingFace Hub. Compute nodes may have **no outbound internet**, so prefetch into a
`/gpfs` cache on the login node:

```bash
module load conda && conda activate $ALLOC/env
export HF_HOME=$ALLOC/hf_cache
python -c "import timm; timm.create_model('resnet50.a1_in1k', pretrained=True)"
# (ResNet50 weights are ~100 MB; this populates $HF_HOME so the job runs offline.)
```

The job script sets the same `HF_HOME` and `HF_HUB_OFFLINE=1`, so the compute node
reads from this cache instead of the network.

---

## 6. Verify (quick, login node)

```bash
cd $ALLOC/car-damage-severity-pipeline
ls data_quality/clean_train_manifest.csv data_quality/heldout_test_manifest.csv
ls $SCRUBBED/data/training $SCRUBBED/data/validation
ls $ALLOC/hf_cache                                   # weight cache populated
python -c "import torch, timm, sklearn, pandas; print('imports OK')"
```

---

## 7. (Optional) Interactive smoke test — within the free tier

Sanity-check env + data + GPU on a few steps before the real submission:

```bash
salloc --account=macsvlarobotics --qos=interactive --gpus=1 --time=00:30:00
# ---- now on a compute node ----
module load conda && conda activate $ALLOC/env
export HF_HOME=$ALLOC/hf_cache HF_HUB_OFFLINE=1
cd $ALLOC/car-damage-severity-pipeline
ln -sfn $SCRUBBED/data/training training && ln -sfn $SCRUBBED/data/validation validation
python tools/compare_resnet_variants.py \
  --train-csv data_quality/clean_train_manifest.csv \
  --test-csv  data_quality/heldout_test_manifest.csv \
  --output-dir $SCRUBBED/artifacts/smoke \
  --variants resnet50 --epochs 1 --batch-size 64
exit    # frees the GPU
```

If the epoch runs and writes `resnet50_results.json`, you're ready to submit.

---

## 8. Submit the training job

The ready-to-use job script is `scripts/slurm/train_resnet50_tillicum.slurm`. It runs
**only ResNet50** (via the `--variants resnet50` flag), 12 epochs, batch size 64, on
1 GPU.

```bash
cd $ALLOC/car-damage-severity-pipeline
sbatch scripts/slurm/train_resnet50_tillicum.slurm
squeue -u $USER
```

What the script does (see the file for the exact directives):

- `--account=macsvlarobotics --qos=normal --gpus=1 --cpus-per-task=8 --mem=64G --time=02:00:00`
- Sets `HF_HOME` + `HF_HUB_OFFLINE=1` so weights load from the `/gpfs` cache.
- `module load conda` + `conda activate $ALLOC/env` (no `cuda` module at runtime).
- `cd`s into the repo and symlinks the staged `training/` and `validation/` folders in.
- Writes results to `$SCRUBBED/artifacts/resnet50/` and logs to `$SCRUBBED/logs/`.

**To train all three variants** for a full comparison instead, drop the
`--variants resnet50` line (the script defaults to resnet18 + resnet34 + resnet50) and
bump `--time` a little. **To change epochs/batch size**, edit those flags in the script.

---

## 9. Monitor & cost

```bash
squeue -u $USER                                          # queue / running
tail -f /gpfs/scrubbed/$USER/car-damage/logs/resnet50_*.out
hyakusage                                                # GPU-hr + cost this cycle
seff <jobid>                                             # per-job GPU/mem efficiency
```

- 1 GPU = **1 GPU-hr per wall-clock hour**. This job finishes in **minutes**, costing a
  small fraction of a GPU-hr — comfortably inside the free 100 GPU-hr tier.
- **Maintenance: 2nd Tuesday/month** — avoid submitting right before it.

---

## 10. Retrieve results (scrubbed has no backup)

`/gpfs/scrubbed` is purged after 60 days idle and never backed up — copy results off:

```bash
# Run from your own machine:
rsync -avz <netid>@tillicum.hyak.uw.edu:/gpfs/scrubbed/<netid>/car-damage/artifacts/resnet50/ \
  ./artifacts/resnet50/
```

Then generate the analysis report locally:

```bash
python tools/analyze_resnet_comparison.py \
  --results-dir artifacts/resnet50 \
  --output-report artifacts/resnet50/analysis_report.md
```

Outputs in `artifacts/resnet50/`: `resnet50_results.json` (accuracy, macro-F1, per-class
precision/recall/F1, confusion matrix, AIC/BIC, inference time) and, if you trained more
than one variant, `comparison_summary.json`.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `dropna` leaves 0 rows / "Train: 0 \| Val: 0" | Image folders not found. Confirm `training/` and `validation/` (with `01-minor` etc. subfolders) are staged on scrubbed and symlinked into the repo CWD. The job runs from the repo root for this reason. |
| Hang or error fetching weights | Compute node has no internet. Re-run the §5 prefetch on the **login** node; confirm `HF_HOME` matches and `HF_HUB_OFFLINE=1` is set. |
| `CUDA error` / `device-side assert` / no GPU | You requested no GPU, or loaded the `cuda` module (cuDNN shadowing). Keep `--gpus=1` and load **only** `conda`. Check `nvidia-smi` in the log. |
| Out of memory | Lower `--batch-size` (e.g. 32). The H200 has 141 GB, so this is unlikely for ResNet50 at 224². |
| Job pending a long time | `squeue` shows the reason; `interactive`/`debug` QOS schedule faster for the smoke test. |

---

## Reference: key paths & commands

| Item | Value |
|---|---|
| SSH | `ssh <netid>@tillicum.hyak.uw.edu` |
| Account | `macsvlarobotics` (`#SBATCH --account=macsvlarobotics`) |
| Conda env | `/gpfs/projects/macsvlarobotics/car-damage/env` (`module load conda` first) |
| Repo | `/gpfs/projects/macsvlarobotics/car-damage/car-damage-severity-pipeline` |
| Dataset | `/gpfs/scrubbed/<netid>/car-damage/data/{training,validation}` |
| Weight cache | `/gpfs/projects/macsvlarobotics/car-damage/hf_cache` (`HF_HOME`) |
| Submit | `sbatch scripts/slurm/train_resnet50_tillicum.slurm` |
| Results | `/gpfs/scrubbed/<netid>/car-damage/artifacts/resnet50` |
| Monitor | `squeue -u $USER`, `hyakusage`, `seff <jobid>` |
