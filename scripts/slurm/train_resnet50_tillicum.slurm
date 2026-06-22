#!/bin/bash
# Train the bigger ResNet (ResNet50) on Tillicum (UW Hyak, H200 SLURM cluster).
# Submit with:  sbatch scripts/slurm/train_resnet50_tillicum.slurm
# See docs/TILLICUM_TRAINING.md for the full setup, staging, and weight-cache steps.

#SBATCH --job-name=cardmg-resnet50
#SBATCH --account=macsvlarobotics
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scrubbed/%u/car-damage/logs/resnet50_%j.out
#SBATCH --error=/gpfs/scrubbed/%u/car-damage/logs/resnet50_%j.err

set -euo pipefail

# ---- paths (edit ALLOC if your allocation differs) ----
ALLOC=/gpfs/projects/macsvlarobotics/car-damage
SCRUBBED=/gpfs/scrubbed/$USER/car-damage
REPO=$ALLOC/car-damage-severity-pipeline
ENV=$ALLOC/env

# ---- offline weight cache (pre-populated on the login node; see docs §5) ----
# timm pulls pretrained weights from HuggingFace Hub. Compute nodes may have NO
# internet, so we point at a /gpfs cache and force offline mode.
export HF_HOME=$ALLOC/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---- environment (torch ships its own CUDA libs; do NOT load the cuda module) ----
module load conda
conda activate "$ENV"

# ---- run from the repo root so the manifest path-resolver finds the images ----
cd "$REPO"

# The manifest CSVs store foreign absolute paths; the script falls back to
# searching ./training and ./validation by filename. Symlink the staged image
# folders (kept on scrubbed) into the repo CWD so that fallback works.
ln -sfn "$SCRUBBED/data/training"   training
ln -sfn "$SCRUBBED/data/validation" validation

mkdir -p "$SCRUBBED/artifacts/resnet50"

nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python tools/compare_resnet_variants.py \
  --train-csv data_quality/clean_train_manifest.csv \
  --test-csv  data_quality/heldout_test_manifest.csv \
  --output-dir "$SCRUBBED/artifacts/resnet50" \
  --variants resnet50 \
  --epochs 12 \
  --batch-size 64

echo "Done. Results in $SCRUBBED/artifacts/resnet50"
