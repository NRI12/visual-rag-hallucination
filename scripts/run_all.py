"""
End-to-end Visual-RAG pipeline — single entry point.

Steps:
  1. Download datasets (POPE, HallusionBench, COCO, Visual Genome)
  2. Build FAISS index from Visual Genome
  3. Run LLaVA baseline evaluation
  4. Run LLaVA + Visual-RAG evaluation
  5. Compare results & generate plots

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --skip_download   # if data already present
    python scripts/run_all.py --max_facts 500000 --max_samples 200
"""
import argparse
import os
import sys
import logging
import time

# Make visual_rag importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visual_rag.utils import setup_logging, load_config, set_seed

log = logging.getLogger(__name__)


def step(msg: str):
    bar = "=" * 55
    print(f"\n{bar}\n {msg}\n{bar}")


def run_download(data_dir: str, skip_coco: bool = False, skip_vg: bool = False):
    from scripts.download_data import (
        download_pope, download_hallusionbench,
        download_coco_val2014, download_visual_genome,
    )
    failed = []
    for name, fn in [
        ("POPE",           lambda: download_pope(data_dir)),
        ("HallusionBench", lambda: download_hallusionbench(data_dir)),
        ("COCO val2014",   lambda: (skip_coco or download_coco_val2014(data_dir))),
        ("Visual Genome",  lambda: (skip_vg  or download_visual_genome(data_dir))),
    ]:
        print(f"\n[{name}]")
        if not fn():
            failed.append(name)

    if failed:
        log.error(f"Dataset download failed: {failed}")
        sys.exit(1)


def run_build_index(cfg: dict, max_facts: int):
    from visual_rag.retrieval import CLIPEncoder, SceneGraphIndexer, load_visual_genome_facts
    import random

    index_path    = cfg["retrieval"]["index_path"]
    metadata_path = cfg["retrieval"]["metadata_path"]

    if os.path.exists(index_path):
        log.info(f"Index already exists at {index_path}, skipping build.")
        return

    encoder = CLIPEncoder(
        cfg["model"]["clip_model"],
        cfg["model"]["clip_pretrained"],
        cfg["model"]["device"],
    )

    facts = load_visual_genome_facts(cfg["data"]["visual_genome_dir"])
    if max_facts and len(facts) > max_facts:
        log.info(f"Capping {len(facts):,} → {max_facts:,} facts")
        random.shuffle(facts)
        facts = facts[:max_facts]

    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    indexer = SceneGraphIndexer(encoder, dim=512)
    indexer.build(facts, batch_size=2048,
                  index_path=index_path,
                  metadata_path=metadata_path,
                  num_workers=4)


def run_baseline(cfg: dict):
    from visual_rag.models import LLaVABaseline
    from visual_rag.data import load_pope_all_splits, HallusionBenchDataset
    from visual_rag.evaluation import POPEEvaluator, HallusionEvaluator

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
    ev = POPEEvaluator(model, output_dir=cfg["evaluation"]["output_dir"])
    results = {}
    for split, ds in pope_splits.items():
        results[split] = ev.run(ds, split, model_tag="baseline")

    hb = HallusionBenchDataset(
        cfg["data"]["hallusionbench_dir"],
        max_samples=cfg["data"]["max_samples"],
    )
    hev = HallusionEvaluator(model, output_dir=cfg["evaluation"]["output_dir"])
    results["hallusionbench"] = hev.run(hb, model_tag="baseline")

    return results


def run_rag(cfg: dict):
    from visual_rag.retrieval import CLIPEncoder, SceneGraphIndexer, VisualRetriever
    from visual_rag.models import LLaVAWithRAG
    from visual_rag.data import load_pope_all_splits, HallusionBenchDataset
    from visual_rag.evaluation import POPEEvaluator, HallusionEvaluator

    encoder = CLIPEncoder(
        cfg["model"]["clip_model"],
        cfg["model"]["clip_pretrained"],
        cfg["model"]["device"],
    )
    indexer = SceneGraphIndexer(encoder)
    indexer.load(cfg["retrieval"]["index_path"],
                 cfg["retrieval"]["metadata_path"])

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
    ev = POPEEvaluator(model, output_dir=cfg["evaluation"]["output_dir"])
    results = {}
    for split, ds in pope_splits.items():
        results[split] = ev.run(ds, split, model_tag="rag")

    hb = HallusionBenchDataset(
        cfg["data"]["hallusionbench_dir"],
        max_samples=cfg["data"]["max_samples"],
    )
    hev = HallusionEvaluator(model, output_dir=cfg["evaluation"]["output_dir"])
    results["hallusionbench"] = hev.run(hb, model_tag="rag")

    return results


def print_summary(baseline: dict, rag: dict):
    print("\n" + "=" * 55)
    print(" FINAL RESULTS")
    print("=" * 55)
    print(f"{'Benchmark':<22} {'Baseline':>10} {'RAG':>10} {'Delta':>8}")
    print("-" * 55)
    for key in ["adversarial", "popular", "random"]:
        b = baseline.get(key, {})
        r = rag.get(key, {})
        if b and r:
            delta = r["accuracy"] - b["accuracy"]
            sign = "+" if delta >= 0 else ""
            print(f"POPE {key:<17} {b['accuracy']:>9.1f}% {r['accuracy']:>9.1f}% "
                  f"{sign}{delta:.1f}%")
    bh = baseline.get("hallusionbench", {})
    rh = rag.get("hallusionbench", {})
    if bh and rh:
        delta = rh["accuracy"] - bh["accuracy"]
        sign = "+" if delta >= 0 else ""
        print(f"{'HallusionBench':<22} {bh['accuracy']:>9.1f}% {rh['accuracy']:>9.1f}% "
              f"{sign}{delta:.1f}%")
    print("=" * 55)


def parse_args():
    p = argparse.ArgumentParser(description="Visual-RAG end-to-end pipeline")
    p.add_argument("--config",        default="configs/default.yaml")
    p.add_argument("--skip_download", action="store_true")
    p.add_argument("--skip_index",    action="store_true")
    p.add_argument("--skip_baseline", action="store_true")
    p.add_argument("--skip_rag",      action="store_true")
    p.add_argument("--skip_coco",     action="store_true",
                   help="Skip COCO download (if images already present)")
    p.add_argument("--max_facts",     type=int, default=1_000_000)
    p.add_argument("--max_samples",   type=int, default=None,
                   help="Override max_samples from config")
    return p.parse_args()


def main():
    args  = parse_args()
    cfg   = load_config(args.config)
    setup_logging(**cfg.get("logging", {}))
    set_seed(42)

    if args.max_samples is not None:
        cfg["data"]["max_samples"] = args.max_samples

    t0 = time.time()

    # ── 1. Download ──────────────────────────────────────────
    if not args.skip_download:
        step("Step 1/5 — Downloading datasets")
        run_download(
            data_dir=cfg["data"].get("data_root", "data"),
            skip_coco=args.skip_coco,
        )

    # ── 2. Build index ───────────────────────────────────────
    if not args.skip_index:
        step("Step 2/5 — Building FAISS index")
        run_build_index(cfg, max_facts=args.max_facts)

    # ── 3. Baseline ──────────────────────────────────────────
    baseline_results = {}
    if not args.skip_baseline:
        step("Step 3/5 — Baseline evaluation (LLaVA, no RAG)")
        baseline_results = run_baseline(cfg)

    # ── 4. RAG ───────────────────────────────────────────────
    rag_results = {}
    if not args.skip_rag:
        step("Step 4/5 — RAG evaluation (LLaVA + Visual-RAG)")
        rag_results = run_rag(cfg)

    # ── 5. Compare ───────────────────────────────────────────
    step("Step 5/5 — Comparing results")
    os.makedirs(cfg["evaluation"]["output_dir"], exist_ok=True)

    if baseline_results and rag_results:
        print_summary(baseline_results, rag_results)

    # Generate plots
    from scripts.compare_results import main as compare_main
    import sys as _sys
    _sys.argv = ["compare_results.py",
                 "--results_dir", cfg["evaluation"]["output_dir"]]
    compare_main()

    elapsed = (time.time() - t0) / 60
    print(f"\nTotal time: {elapsed:.1f} min")
    print(f"Results in: {os.path.abspath(cfg['evaluation']['output_dir'])}/")


if __name__ == "__main__":
    main()
