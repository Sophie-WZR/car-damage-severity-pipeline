#!/usr/bin/env python3
"""
Pre-download timm pretrained weights into the HuggingFace cache.

Run this on a node WITH internet (on Tillicum: the LOGIN node). Tillicum compute
nodes may have no outbound internet, so the training job runs with HF_HUB_OFFLINE=1
and reads from this cache instead of hitting the network.

    export HF_HOME=/gpfs/projects/macsvlarobotics/car-damage/hf_cache
    python tools/prefetch_weights.py --variants resnet50 resnet101 resnet152

With no --variants, prefetches every architecture in the registry. The variant->tag
mapping comes from compare_resnet_variants.ARCH, so this never drifts from training.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_resnet_variants import ARCH  # noqa: E402
import timm  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Prefetch timm pretrained weights into HF_HOME cache")
    p.add_argument("--variants", nargs="+", default=list(ARCH), choices=list(ARCH),
                   help="Variants to cache (default: all registered).")
    args = p.parse_args()

    import os
    print(f"HF_HOME = {os.environ.get('HF_HOME', '(default ~/.cache/huggingface)')}")
    for v in args.variants:
        arch = ARCH[v]
        print(f"  prefetch {v:10s} ({arch}) ...", flush=True)
        # pretrained=True triggers the download into the HF cache; we discard the model.
        timm.create_model(arch, pretrained=True)
    print("done — cache populated. Set HF_HUB_OFFLINE=1 in the job to use it offline.")


if __name__ == "__main__":
    main()
