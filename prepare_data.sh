#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${SFT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" sft/scripts/prepare_code_sft_data.py \
  --source_dir "${SOURCE_DIR:-python_code_instructions_18k_alpaca}" \
  --output_dir "${OUTPUT_DIR:-sft/data}" \
  --limit "${LIMIT:-0}" \
  --train_ratio "${TRAIN_RATIO:-0.90}" \
  --valid_ratio "${VALID_RATIO:-0.05}" \
  --seed "${SEED:-42}"
