#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TARGET="${1:-}"
case "$TARGET" in
  l4|L4)
    RESULT_DIR="results/L4-sm89"
    MESSAGE="Add L4 Triton pipeline dumps"
    ;;
  h100|H100)
    RESULT_DIR="results/H100-sm90"
    MESSAGE="Add H100 Triton pipeline dumps"
    ;;
  *)
    echo "Usage: bash scripts/commit_results.sh {l4|h100}" >&2
    exit 2
    ;;
esac

if [[ ! -d "$RESULT_DIR" ]]; then
  echo "Result directory not found: $RESULT_DIR" >&2
  exit 1
fi

# Lightning Studios can start without Git author identity. Keep it repo-local.
if [[ -z "$(git config --get user.name || true)" ]]; then
  git config user.name "Yunseo Lee"
fi
if [[ -z "$(git config --get user.email || true)" ]]; then
  git config user.email "lys139011@gmail.com"
fi

# Preserve the environment snapshot if setup_lightning.sh created it.
if [[ -f environment/version.txt ]]; then
  git add environment/version.txt
fi

# .gitignore intentionally excludes binary *.cubin; textual compiler artifacts
# (TTIR/TTGIR/LLVM IR/PTX/SASS/JSON) are staged.
git add "$RESULT_DIR"

if git diff --cached --quiet; then
  echo "No new staged result changes for $TARGET."
  exit 0
fi

echo "Staged files:"
git diff --cached --name-status

git commit -m "$MESSAGE"
git push origin main

echo
echo "Published $TARGET results to origin/main."
