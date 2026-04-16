"""
Step 3: Run LLaVA + Visual-RAG on POPE + HallusionBench.
Usage:
    python scripts/run_rag.py --config configs/default.yaml
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval import CLIPEncoder, SceneGraphIndexer, VisualRetriever
from src.models import LLaVAWithRAG
from src.data import load_pope_all_splits, HallusionBenchDataset
from src.evaluation import POPEEvaluator, HallusionEvaluator
from src.utils import setup_logging, load_config, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(**cfg.get("logging", {}))
    set_seed(42)

    # Build retriever
    print("Loading CLIP encoder...")
    encoder = CLIPEncoder(
        cfg["model"]["clip_model"],
        cfg["model"]["clip_pretrained"],
        cfg["model"]["device"],
    )
    indexer = SceneGraphIndexer(encoder)
    index_path = cfg["retrieval"]["index_path"]
    metadata_path = cfg["retrieval"]["metadata_path"]

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Run scripts/build_index.py first."
        )
    indexer.load(index_path, metadata_path)

    retriever = VisualRetriever(
        encoder, indexer,
        top_k=cfg["retrieval"]["top_k"],
        score_threshold=cfg["retrieval"]["score_threshold"],
    )

    model = LLaVAWithRAG(
        model_name=cfg["model"]["vlm_name"],
        retriever=retriever,
        device=cfg["model"]["device"],
        dtype=cfg["model"]["dtype"],
    )

    pope_splits = load_pope_all_splits(
        cfg["data"]["pope_dir"],
        cfg["data"]["coco_dir"],
        max_samples=cfg["data"]["max_samples"],
    )
    evaluator = POPEEvaluator(model, output_dir=cfg["evaluation"]["output_dir"])
    all_metrics = {}
    for split_name, dataset in pope_splits.items():
        metrics = evaluator.run(dataset, split_name, model_tag="rag")
        all_metrics[split_name] = metrics

    hallusion = HallusionBenchDataset(
        cfg["data"]["hallusionbench_dir"],
        max_samples=cfg["data"]["max_samples"],
    )
    h_evaluator = HallusionEvaluator(model, output_dir=cfg["evaluation"]["output_dir"])
    h_metrics = h_evaluator.run(hallusion, model_tag="rag")

    print("\n=== RAG RESULTS ===")
    for split, m in all_metrics.items():
        print(f"POPE {split:12s}: Acc={m['accuracy']}%  F1={m['f1']}%")
    print(f"HallusionBench     : Acc={h_metrics['accuracy']}%")


if __name__ == "__main__":
    main()
