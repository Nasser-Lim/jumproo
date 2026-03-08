# -*- coding: utf-8 -*-
"""
V8 Full Training Pipeline — Run on OBS 11 (RTX 4060)

Step 1: Create dataset (8-channel features + sector IDs + regression targets)
Step 2: Train PatchTST Regressor

Usage:
  cd c:/Users/user/source/repos/jumproo
  python -X utf8 stock_prediction_v8/scripts/run_training.py

Prerequisites:
  - Raw CSV data in stock_prediction/data/raw/ (647+ files)
  - market_index.csv already downloaded (run download_market_index.py)
  - sector_map.json already created (run crawl_sectors.py)
  - PyTorch with CUDA support installed
"""
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def check_environment():
    """Verify GPU and dependencies."""
    print("=" * 60)
    print("  V8 Training Environment Check")
    print("=" * 60)

    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  CUDA    : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM    : {vram:.1f} GB")
    else:
        print("  WARNING: CUDA not available. Training on CPU will be very slow!")

    root = Path(__file__).parent.parent
    market_path = root / "data" / "market_index.csv"
    sector_path = root / "configs" / "sector_map.json"
    raw_dir = root.parent / "stock_prediction" / "data" / "raw"

    print(f"\n  Market index : {'OK' if market_path.exists() else 'MISSING — run download_market_index.py'}")
    print(f"  Sector map   : {'OK' if sector_path.exists() else 'MISSING — run crawl_sectors.py'}")

    csv_count = len(list(raw_dir.glob("*.csv"))) if raw_dir.exists() else 0
    print(f"  Raw CSVs     : {csv_count}")
    print("=" * 60)

    if not market_path.exists() or not sector_path.exists():
        print("\nERROR: Missing prerequisite files. Run the scripts above first.")
        sys.exit(1)

    return True


def main():
    check_environment()

    config_path = Path(__file__).parent.parent / "configs" / "v8_config.yaml"

    # Step 1: Create dataset
    print("\n" + "=" * 60)
    print("  STEP 1: Creating V8 Dataset")
    print("=" * 60 + "\n")
    t0 = time.time()

    from src.data.create_dataset_v8 import create_dataset
    create_dataset(config_path)

    print(f"\n  Dataset creation took {(time.time() - t0)/60:.1f} min")

    # Step 2: Train model
    print("\n" + "=" * 60)
    print("  STEP 2: Training PatchTST Regressor")
    print("=" * 60 + "\n")
    t1 = time.time()

    from src.train.train_v8 import train
    train(config_path)

    print(f"\n  Training took {(time.time() - t1)/60:.1f} min")
    print(f"  Total pipeline: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
