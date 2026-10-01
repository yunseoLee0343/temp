#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p environment
python - <<'PY' > environment/version.txt
import sys
import triton
print("python:", sys.version.replace("\n", " "))
print("triton:", triton.__version__)
print("triton_path:", triton.__file__)
PY

cat environment/version.txt
echo
echo "Setup complete. Activate with: source .venv/bin/activate"
