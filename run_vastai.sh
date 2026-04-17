#!/bin/bash
# ============================================================
# Visual-RAG Hallucination — Vast.ai / Remote GPU Runner
# Usage:
#   bash run_vastai.sh                    # full pipeline
#   bash run_vastai.sh --skip_download    # skip if data exists
#   bash run_vastai.sh --max_samples 200  # quick test run
# ============================================================
set -uo pipefail

REPO_URL="https://github.com/NRI12/visual-rag-hallucination.git"
WORKDIR="/workspace/visual-rag-hallucination"

echo "================================================"
echo " Visual-RAG Hallucination — Full Pipeline"
echo "================================================"

# 1. Clone or update repo
if [ -d "$WORKDIR/.git" ]; then
    echo "[git] Pulling latest..."
    cd "$WORKDIR" && git pull
else
    echo "[git] Cloning repo..."
    git clone "$REPO_URL" "$WORKDIR"
    cd "$WORKDIR"
fi

# 2. Install Python package (editable)
echo "[pip] Installing dependencies..."
pip install -q --root-user-action=ignore -r requirements.txt
pip install -q --root-user-action=ignore -e .

# 3. Run full pipeline via Python
echo "[run] Starting end-to-end pipeline..."
python scripts/run_all.py "$@"
