#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/default.yaml}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

if ! python3 - <<'PY' >/dev/null 2>&1
import numpy  # noqa: F401
import pandas  # noqa: F401
import torch  # noqa: F401
import yaml  # noqa: F401
import sklearn  # noqa: F401
import openpyxl  # noqa: F401
PY
then
  echo "Missing Python dependencies. Please run:"
  echo "  python3 -m pip install -e ."
  exit 1
fi

python3 "$ROOT_DIR/scripts/train_stage2.py" --config "$CONFIG_PATH"

echo "Transformer attack-type classifier training complete."
