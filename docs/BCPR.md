# BCPR training and inference

This document describes the runnable BCPR implementation in this repository.
Run every command from the repository root.

## 1. Method path

For each current detection query and each visible historical observation, BCPR
produces a row-centered scalar bias. The bias is repeated over the decoder attention
heads and added to MOTIP's cross-attention mask. Padding and causal masks remain
authoritative.

BCPR training evaluates the same query, history, target columns, and decoder parameters
twice:

```text
OFF endpoint: BCPR alpha = 0
ON endpoint : BCPR alpha = BCPR_ON_ALPHA
ID objective: BCPR_OFF_LOSS_WEIGHT * L_OFF
            + BCPR_ON_LOSS_WEIGHT  * L_ON
```

The default configs use equal endpoint weights and an ON alpha of 1.0. Inference alpha
is independent and can be supplied with `--bcpr-alpha`; the reported BFT setting is
0.75.

## 2. Installation

Follow the upstream [installation guide](./INSTALL.md), including compilation of the
multi-scale deformable-attention operator. The commands below assume the resulting
environment is active and provides `python`, `accelerate`, PyTorch, and TrackEval's
dependencies.

## 3. Data and detector pretraining

Use the upstream dataset structure described in [DATASET.md](./DATASET.md). The relevant
paths are:

```text
datasets/
├── DanceTrack/
└── BFT/

pretrains/
├── r50_deformable_detr_coco_dancetrack.pth
└── r50_deformable_detr_coco_bft.pth
```

The association decoder, trajectory model, and BCPR module start from scratch. Only the
dataset-specific detector pretraining file is loaded. Training configs deliberately keep
`RESUME_MODEL` empty.

## 4. Regression tests

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/test_bcpr_core.py tests/test_bcpr_decoder.py
```

Expected result: all tests pass. The default BCPR module has 45,378 trainable
parameters and is an exact no-op at initialization.

Check the selected config, local dataset/pretrain paths, parameter count, and optionally
an existing checkpoint before launching a long run:

```bash
CUDA_VISIBLE_DEVICES='' python tools/check_bcpr_setup.py \
  --config-path configs/bcpr_bft.yaml
```

```bash
CUDA_VISIBLE_DEVICES='' python tools/check_bcpr_setup.py \
  --config-path configs/bcpr_bft.yaml \
  --checkpoint outputs/bcpr_bft_seed42/checkpoint_21.pth
```

## 5. Training

### BFT

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 train.py \
  --config-path configs/bcpr_bft.yaml
```

Outputs are written to `outputs/bcpr_bft_seed42/`. Checkpoints are named
`checkpoint_<epoch>.pth`, and the text log is
`outputs/bcpr_bft_seed42/train/log.txt`.

### DanceTrack

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 train.py \
  --config-path configs/bcpr_dancetrack.yaml
```

Outputs are written to `outputs/bcpr_dancetrack_seed42/`.

### Monitor training

```bash
tail -F outputs/bcpr_bft_seed42/train/log.txt \
  | grep --line-buffered -E '\[Metrics\]|bcpr/'
```

```bash
watch -n 1 nvidia-smi
```

## 6. Evaluation and inference

`--inference-model` is mandatory in practice; the public configs intentionally do not
contain a private checkpoint path.

### BFT test, alpha 0.75

```bash
CUDA_VISIBLE_DEVICES=0 python submit_and_evaluate.py \
  --config-path configs/bcpr_bft_eval.yaml \
  --inference-model outputs/bcpr_bft_seed42/checkpoint_21.pth \
  --bcpr-alpha 0.75
```

### BFT OFF and ON endpoint comparison

```bash
CUDA_VISIBLE_DEVICES=0 python submit_and_evaluate.py \
  --config-path configs/bcpr_bft_eval.yaml \
  --inference-model outputs/bcpr_bft_seed42/checkpoint_21.pth \
  --bcpr-alpha 0.0 --inference-group bcpr_off
```

```bash
CUDA_VISIBLE_DEVICES=0 python submit_and_evaluate.py \
  --config-path configs/bcpr_bft_eval.yaml \
  --inference-model outputs/bcpr_bft_seed42/checkpoint_21.pth \
  --bcpr-alpha 0.75 --inference-group bcpr_on_alpha075
```

### DanceTrack validation

```bash
CUDA_VISIBLE_DEVICES=0 python submit_and_evaluate.py \
  --config-path configs/bcpr_dancetrack_eval.yaml \
  --inference-model outputs/bcpr_dancetrack_seed42/checkpoint_9.pth \
  --bcpr-alpha 0.75
```

For a benchmark test split without public ground truth, set `INFERENCE_MODE: submit` in
a derived config or pass `--inference-mode submit`. Do not claim test metrics unless the
benchmark evaluator or local ground truth actually provides them.

## 7. Resume behavior

Fresh configs do not auto-resume. To resume explicitly, pass a BCPR checkpoint and the
desired state flags, for example:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 train.py \
  --config-path configs/bcpr_bft.yaml \
  --resume-model outputs/bcpr_bft_seed42/checkpoint_10.pth \
  --resume-optimizer True --resume-scheduler True
```

Use a different `--outputs-dir` and `--exp-name` when the resumed run should not append
to the existing experiment directory.

## 8. Reproduced BFT reference

The existing internal run used epoch 21 and inference alpha 0.75 and produced:

| HOTA | DetA | AssA | MOTA | IDF1 |
|---:|---:|---:|---:|---:|
| 71.174 | 69.772 | 72.841 | 77.125 | 83.059 |

Checkpoint binaries, datasets, and pretrained detector weights are not redistributed in
this source repository.

## 9. Scope

This public branch contains BCPR only. Later experimental modules such as
G-COVE and QG-COMP are intentionally excluded so that the software-registration artifact
and reproducibility package correspond to the BCPR method alone.
