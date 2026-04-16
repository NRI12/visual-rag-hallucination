#!/bin/bash
# ============================================================
# Visual-RAG Hallucination — Vast.ai Full Pipeline Runner
# Run this on the remote GPU instance via SSH:
#   bash run_vastai.sh
# ============================================================
set -uo pipefail

REPO_URL="https://github.com/NRI12/visual-rag-hallucination.git"
WORKDIR="/workspace/visual-rag-hallucination"
HF_CACHE="/workspace/hf_cache"
DATA_DIR="$WORKDIR/data"

echo "========================================"
echo " Visual-RAG Hallucination — Vast.ai Run"
echo "========================================"

# ---------- 1. System deps ----------
# pytorch/pytorch image already has git, wget, unzip
# Only install aria2 if missing (used for fast parallel downloads)
echo "[1/7] Checking system packages..."
if ! command -v aria2c &>/dev/null; then
    apt-get install -y -qq --no-install-recommends aria2 2>/dev/null || \
    pip install -q aria2  # fallback: use wget if aria2 unavailable
fi

# ---------- 2. Clone repo ----------
echo "[2/7] Cloning repository..."
if [ -d "$WORKDIR" ]; then
    cd "$WORKDIR" && git pull
else
    git clone "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"

# ---------- 3. Python env ----------
echo "[3/7] Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# ---------- 4. Download datasets ----------
echo "[4/7] Downloading datasets..."
mkdir -p "$DATA_DIR/visual_genome" "$DATA_DIR/pope" \
         "$DATA_DIR/hallusionbench" "$DATA_DIR/coco"

# POPE annotations (from GitHub raw)
if [ ! -f "$DATA_DIR/pope/coco_pope_adversarial.json" ]; then
    echo "  Downloading POPE..."
    BASE="https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco"
    wget -q --show-progress -P "$DATA_DIR/pope" \
        "$BASE/coco_pope_adversarial.json" \
        "$BASE/coco_pope_popular.json" \
        "$BASE/coco_pope_random.json" && \
        echo "  POPE OK" || echo "  POPE download failed — skipping"
fi

# HallusionBench
if [ ! -f "$DATA_DIR/hallusionbench/HallusionBench.json" ]; then
    echo "  Downloading HallusionBench..."
    wget --show-progress -O /tmp/hallusionbench.zip \
        "https://github.com/tianyi-lab/HallusionBench/archive/refs/heads/main.zip" && \
    unzip -o /tmp/hallusionbench.zip -d /tmp/ && \
    cp -r /tmp/HallusionBench-main/. "$DATA_DIR/hallusionbench/" && \
    echo "  HallusionBench OK" || echo "  HallusionBench download failed — skipping"
fi

# COCO val2014
if [ ! -d "$DATA_DIR/coco/val2014" ]; then
    echo "  Downloading COCO val2014 (~7GB, may take 10-20 min)..."
    mkdir -p "$DATA_DIR/coco"
    if command -v aria2c &>/dev/null; then
        aria2c -x 8 -s 8 \
            "http://images.cocodataset.org/zips/val2014.zip" \
            -d "$DATA_DIR/coco"
    else
        wget --show-progress -P "$DATA_DIR/coco" \
            "http://images.cocodataset.org/zips/val2014.zip"
    fi
    unzip -o "$DATA_DIR/coco/val2014.zip" -d "$DATA_DIR/coco/" && \
    rm -f "$DATA_DIR/coco/val2014.zip" && \
    echo "  COCO val2014 OK" || echo "  COCO download failed — check disk space"
fi

# Visual Genome relationships + attributes
if [ ! -f "$DATA_DIR/visual_genome/relationships.json" ]; then
    echo "  Downloading Visual Genome..."
    wget --show-progress -P "$DATA_DIR/visual_genome" \
        "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships.json.zip" \
        "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/attributes.json.zip" && \
    unzip -o "$DATA_DIR/visual_genome/relationships.json.zip" -d "$DATA_DIR/visual_genome/" && \
    unzip -o "$DATA_DIR/visual_genome/attributes.json.zip"   -d "$DATA_DIR/visual_genome/" && \
    rm -f "$DATA_DIR/visual_genome"/*.zip && \
    echo "  Visual Genome OK" || echo "  Visual Genome download failed"
fi

# ---------- 5. Build FAISS index ----------
echo "[5/7] Building FAISS index from Visual Genome..."
export HF_HOME="$HF_CACHE"
python scripts/build_index.py \
    --vg_dir "$DATA_DIR/visual_genome" \
    --index_path "$DATA_DIR/vg_faiss.index" \
    --metadata_path "$DATA_DIR/vg_metadata.json" \
    --device cuda

# ---------- 6. Run experiments ----------
echo "[6/7] Running baseline evaluation..."
python scripts/run_baseline.py --config configs/default.yaml

echo "  Running RAG evaluation..."
python scripts/run_rag.py --config configs/default.yaml

# ---------- 7. Compare & plot ----------
echo "[7/7] Generating comparison report..."
python scripts/compare_results.py --results_dir results

echo ""
echo "=============================="
echo " ALL DONE — results/ contains:"
ls -lh results/
echo "=============================="
