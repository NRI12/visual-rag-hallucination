"""
Step 2: Run LLaVA baseline (no RAG) on POPE + HallusionBench.
Usage:
    python scripts/run_baseline.py --config configs/default.yaml
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visual_rag.models import LLaVABaseline
from visual_rag.data import load_pope_all_splits, HallusionBenchDataset
from visual_rag.evaluation import POPEEvaluator, HallusionEvaluator
from visual_rag.utils import setup_logging, load_config, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(**cfg.get("logging", {}))
    set_seed(42)

    model = LLaVABaseline(
        model_name=cfg["model"]["vlm_name"],
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
        metrics = evaluator.run(dataset, split_name, model_tag="baseline")
        all_metrics[split_name] = metrics

    hallusion = HallusionBenchDataset(
        cfg["data"]["hallusionbench_dir"],
        max_samples=cfg["data"]["max_samples"],
    )
    h_evaluator = HallusionEvaluator(model, output_dir=cfg["evaluation"]["output_dir"])
    h_metrics = h_evaluator.run(hallusion, model_tag="baseline")

    print("\n=== BASELINE RESULTS ===")
    for split, m in all_metrics.items():
        print(f"POPE {split:12s}: Acc={m['accuracy']}%  F1={m['f1']}%")
    print(f"HallusionBench     : Acc={h_metrics['accuracy']}%")


if __name__ == "__main__":
    main()
