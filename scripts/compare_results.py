"""
Step 4: Compare baseline vs RAG results and generate plots.
Usage:
    python scripts/compare_results.py --results_dir results
"""
import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import pandas as pd


def load_metrics(results_dir, prefix, model_tag):
    path = os.path.join(results_dir, f"{prefix}_{model_tag}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["metrics"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    args = p.parse_args()

    splits = ["adversarial", "popular", "random"]
    rows = []

    for split in splits:
        for tag in ["baseline", "rag"]:
            m = load_metrics(args.results_dir, f"pope_{split}", tag)
            if m:
                rows.append({
                    "Model": tag.upper(),
                    "Split": split,
                    "Accuracy": m["accuracy"],
                    "F1": m["f1"],
                    "Precision": m["precision"],
                    "Recall": m["recall"],
                    "Yes Rate": m["yes_rate"],
                })

    df = pd.DataFrame(rows)
    print("\n=== POPE Benchmark Comparison ===")
    print(df.to_string(index=False))

    # Add HallusionBench
    hal_rows = []
    for tag in ["baseline", "rag"]:
        m = load_metrics(args.results_dir, "hallusionbench", tag)
        if m:
            hal_rows.append({"Model": tag.upper(), "Accuracy": m["accuracy"]})
    df_hal = pd.DataFrame(hal_rows)
    print("\n=== HallusionBench Comparison ===")
    print(df_hal.to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Visual-RAG vs Baseline: Hallucination Reduction", fontsize=14)

    for i, split in enumerate(splits):
        sub = df[df["Split"] == split]
        if sub.empty:
            continue
        x = sub["Model"].tolist()
        accs = sub["Accuracy"].tolist()
        f1s = sub["F1"].tolist()
        x_pos = range(len(x))

        axes[i].bar([p - 0.2 for p in x_pos], accs, width=0.35, label="Accuracy", alpha=0.8)
        axes[i].bar([p + 0.2 for p in x_pos], f1s, width=0.35, label="F1", alpha=0.8)
        axes[i].set_xticks(list(x_pos))
        axes[i].set_xticklabels(x)
        axes[i].set_ylim(0, 100)
        axes[i].set_title(f"POPE {split}")
        axes[i].legend()
        axes[i].set_ylabel("Score (%)")

    plt.tight_layout()
    out = os.path.join(args.results_dir, "comparison_plot.png")
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved → {out}")

    # Delta table
    if len(df) > 0:
        delta_rows = []
        for split in splits:
            base = df[(df["Split"] == split) & (df["Model"] == "BASELINE")]
            rag = df[(df["Split"] == split) & (df["Model"] == "RAG")]
            if base.empty or rag.empty:
                continue
            delta_rows.append({
                "Split": split,
                "ΔAccuracy": round(rag["Accuracy"].values[0] - base["Accuracy"].values[0], 2),
                "ΔF1": round(rag["F1"].values[0] - base["F1"].values[0], 2),
            })
        print("\n=== Delta (RAG - Baseline) ===")
        print(pd.DataFrame(delta_rows).to_string(index=False))


if __name__ == "__main__":
    main()
