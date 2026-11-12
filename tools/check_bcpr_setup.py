"""Preflight check for BCPR training and inference.

This command does not start training and does not modify checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from configs.util import load_super_config
from models.misc import load_checkpoint
from models.motip import build as build_motip
from utils.misc import yaml_to_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default="configs/bcpr_bft.yaml")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()

    raw = yaml_to_dict(args.config_path)
    config = load_super_config(raw, raw["SUPER_CONFIG_PATH"])
    config["DEVICE"] = "cpu"

    assert config.get("USE_BCPR") is True, "USE_BCPR must be enabled."
    assert config.get("BCPR_DUAL_PATH") is True, "BCPR dual-path training is disabled."
    assert abs(config["BCPR_OFF_LOSS_WEIGHT"] + config["BCPR_ON_LOSS_WEIGHT"] - 1.0) < 1e-12

    model, _ = build_motip(config)
    bcpr = model.id_decoder.bcpr
    assert bcpr is not None
    assert bcpr.parameter_count == 45_378

    detector = Path(config["DETR_PRETRAIN"])
    dataset_paths = [Path(config["DATA_ROOT"]) / name for name in config["DATASETS"]]

    print(f"config: {args.config_path}")
    print(f"datasets: {config['DATASETS']} / splits: {config['DATASET_SPLITS']}")
    for dataset in dataset_paths:
        print(f"dataset path: {dataset} ({'OK' if dataset.exists() else 'MISSING'})")
    print(f"detector pretrain: {detector} ({'OK' if detector.exists() else 'MISSING'})")
    print(f"BCPR module: {type(bcpr).__name__}")
    print(f"BCPR parameters: {bcpr.parameter_count:,}")
    print(f"dual-path weights: OFF={config['BCPR_OFF_LOSS_WEIGHT']}, "
          f"ON={config['BCPR_ON_LOSS_WEIGHT']}, alpha={config['BCPR_ON_ALPHA']}")
    print(f"output directory: {config['OUTPUTS_DIR']}")

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        load_checkpoint(model, str(checkpoint_path))
        print(f"checkpoint strict load: OK ({checkpoint_path})")

    missing = [str(path) for path in [*dataset_paths, detector] if not path.exists()]
    if missing:
        print("preflight: model/config OK; local data or pretrain files are missing")
        raise SystemExit(2)
    print("preflight: OK")


if __name__ == "__main__":
    main()
