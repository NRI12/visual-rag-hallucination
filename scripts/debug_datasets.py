"""
Quick sanity check for datasets before running full eval.
Usage:
    python scripts/debug_datasets.py --config configs/default.yaml
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from visual_rag.utils import load_config, setup_logging

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    setup_logging()

    # ── POPE ────────────────────────────────────────────────
    print("\n[POPE]")
    try:
        from visual_rag.data import load_pope_all_splits
        splits = load_pope_all_splits(cfg["data"]["pope_dir"],
                                      cfg["data"]["coco_dir"], max_samples=5)
        for name, ds in splits.items():
            item = ds[0]
            print(f"  {name}: {len(ds)} samples | "
                  f"img={item['image'].size} | "
                  f"q='{item['question'][:50]}' | label={item['label']}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── HallusionBench ───────────────────────────────────────
    print("\n[HallusionBench]")
    try:
        from visual_rag.data import HallusionBenchDataset
        ds = HallusionBenchDataset(cfg["data"]["hallusionbench_dir"], max_samples=10)
        has_img = sum(1 for i in range(len(ds)) if ds[i]["image"] is not None)
        print(f"  Total: {len(ds)} | With images: {has_img}")
        for i in range(min(3, len(ds))):
            item = ds[i]
            print(f"  [{i}] gt={item['gt_answer']} | "
                  f"img={'OK' if item['image'] else 'MISSING'} | "
                  f"q='{item['question'][:50]}'")
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── FAISS index ──────────────────────────────────────────
    print("\n[FAISS Index]")
    index_path = cfg["retrieval"]["index_path"]
    if os.path.exists(index_path):
        size_gb = os.path.getsize(index_path) / 1e9
        print(f"  {index_path}: {size_gb:.2f} GB ✓")
    else:
        print(f"  MISSING: {index_path}")

    # ── Retrieval test ───────────────────────────────────────
    print("\n[Retrieval test]")
    try:
        from visual_rag.retrieval import CLIPEncoder, SceneGraphIndexer, VisualRetriever
        encoder = CLIPEncoder(cfg["model"]["clip_model"],
                              cfg["model"]["clip_pretrained"],
                              cfg["model"]["device"])
        indexer = SceneGraphIndexer(encoder)
        indexer.load(cfg["retrieval"]["index_path"],
                     cfg["retrieval"]["metadata_path"])
        retriever = VisualRetriever(encoder, indexer,
                                    top_k=cfg["retrieval"]["top_k"],
                                    score_threshold=cfg["retrieval"]["score_threshold"])

        from visual_rag.data import load_pope_all_splits
        splits = load_pope_all_splits(cfg["data"]["pope_dir"],
                                      cfg["data"]["coco_dir"], max_samples=1)
        item = splits["adversarial"][0]
        facts = retriever.retrieve(item["image"], item["question"])
        print(f"  Question: {item['question']}")
        print(f"  Retrieved {len(facts)} facts:")
        for f, s in facts:
            print(f"    [{s:.3f}] {f}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()
