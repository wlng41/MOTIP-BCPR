#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 CHECKPOINT [ALPHA]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="$1"
ALPHA="${2:-0.75}"
cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  python submit_and_evaluate.py \
  --config-path configs/bcpr_dancetrack_eval.yaml \
  --inference-model "$CHECKPOINT" \
  --bcpr-alpha "$ALPHA" \
  --inference-group "bcpr_alpha_${ALPHA}"
