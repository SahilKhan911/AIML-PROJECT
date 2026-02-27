#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "[1/4] Running quick checks..."
python3 -m py_compile app.py src/ml_pipeline.py scripts_benchmark.py

echo "[2/4] Verifying required deployment files..."
for f in requirements.txt app.py render.yaml .streamlit/config.toml runtime.txt README.md; do
  if [[ ! -f "$f" ]]; then
    echo "Missing required file: $f"
    exit 1
  fi
done

echo "[3/4] Staging git files..."
git add .

echo "[4/4] Ready to commit/push."
echo "Run these commands next:"
echo "  git commit -m 'Deploy-ready Project 2 Milestone 1 app'"
echo "  git remote add origin <YOUR_GITHUB_REPO_URL>"
echo "  git push -u origin main"

echo
echo "Then deploy from either:"
echo "  - Streamlit Community Cloud (repo + app.py)"
echo "  - Render (connect repo; render.yaml auto-detected)"
