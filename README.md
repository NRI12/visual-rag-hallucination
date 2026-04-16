# Visual-RAG: Retrieval-Augmented Hallucination Grounding for VLMs

> **CVPR/ECCV 2026 submission** — Novel framework reducing hallucination in Vision-Language Models via structured scene-graph retrieval.

## Overview

Current VLMs (LLaVA, InstructBLIP, etc.) hallucinate objects not present in images at rates of 30–40% on adversarial benchmarks. We propose **Visual-RAG**: at inference time, retrieve relevant scene-graph facts from Visual Genome via CLIP-encoded FAISS index, then inject them as grounding context into the VLM prompt.

```
Image + Question
      │
      ▼
  CLIP Encoder ──► FAISS Index (Visual Genome facts)
      │                    │
      └────────────────────┘
             Top-K Facts
                 │
                 ▼
    LLaVA Prompt (question + retrieved context)
                 │
                 ▼
         Grounded Answer ✓
```

## Key Results (expected)

| Benchmark | Baseline | Visual-RAG | Δ |
|-----------|----------|------------|---|
| POPE Adversarial (Acc) | ~84% | ~88% | +4% |
| POPE Popular (Acc) | ~86% | ~90% | +4% |
| POPE Random (Acc) | ~88% | ~91% | +3% |
| HallusionBench (Acc) | ~52% | ~58% | +6% |

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Build FAISS index
```bash
python scripts/build_index.py \
    --vg_dir data/visual_genome \
    --index_path data/vg_faiss.index \
    --metadata_path data/vg_metadata.json
```

### 3. Run baseline
```bash
python scripts/run_baseline.py --config configs/default.yaml
```

### 4. Run Visual-RAG
```bash
python scripts/run_rag.py --config configs/default.yaml
```

### 5. Compare results
```bash
python scripts/compare_results.py --results_dir results
```

## Run on Vast.ai
```bash
# On your Vast.ai instance:
bash run_vastai.sh
```

## Project Structure
```
visual-rag-hallucination/
├── src/
│   ├── retrieval/          # CLIP encoder, FAISS indexer, retriever
│   ├── models/             # LLaVA baseline + RAG variant
│   ├── data/               # POPE, HallusionBench loaders
│   ├── evaluation/         # Metrics (POPE, CHAIR) + evaluators
│   └── utils/              # Logging, config, seeding
├── scripts/
│   ├── build_index.py      # Build VG FAISS index
│   ├── run_baseline.py     # Baseline eval
│   ├── run_rag.py          # RAG eval
│   └── compare_results.py  # Plots & delta table
├── configs/default.yaml
└── run_vastai.sh           # Full pipeline on Vast.ai
```

## Datasets
- **POPE**: [github.com/AoiDragon/POPE](https://github.com/AoiDragon/POPE) — 3×3000 yes/no hallucination questions
- **HallusionBench**: [github.com/tianyi-lab/HallusionBench](https://github.com/tianyi-lab/HallusionBench) — 1,129 visual illusion questions
- **Visual Genome**: [visualgenome.org](https://visualgenome.org/) — 108K images, 2.3M scene graph facts
- **COCO val2014**: standard image source for POPE

## Citation
```bibtex
@misc{visualrag2026,
  title   = {Visual-RAG: Scene-Graph Retrieval for Hallucination Grounding in VLMs},
  year    = {2026},
}
```
