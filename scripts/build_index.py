"""
Step 1: Build FAISS index from Visual Genome scene graphs.
Usage:
    python scripts/build_index.py --vg_dir data/visual_genome \
                                   --index_path data/vg_faiss.index \
                                   --metadata_path data/vg_metadata.json
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visual_rag.retrieval import CLIPEncoder, SceneGraphIndexer, load_visual_genome_facts
from visual_rag.utils import setup_logging, load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vg_dir", default="data/visual_genome")
    p.add_argument("--index_path", default="data/vg_faiss.index")
    p.add_argument("--metadata_path", default="data/vg_metadata.json")
    p.add_argument("--clip_model", default="ViT-B-32")
    p.add_argument("--clip_pretrained", default="openai")
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--max_facts", type=int, default=1000000,
                   help="Cap facts for faster indexing (1M is enough for retrieval)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()

    print("Loading CLIP encoder...")
    encoder = CLIPEncoder(args.clip_model, args.clip_pretrained, args.device)

    print("Loading Visual Genome facts...")
    facts = load_visual_genome_facts(args.vg_dir)
    print(f"  → {len(facts)} facts loaded")

    if args.max_facts and len(facts) > args.max_facts:
        print(f"  Capping to {args.max_facts:,} facts (from {len(facts):,})")
        import random; random.shuffle(facts)
        facts = facts[:args.max_facts]

    print("Building FAISS index...")
    indexer = SceneGraphIndexer(encoder, dim=512)
    indexer.build(facts, batch_size=args.batch_size,
                  index_path=args.index_path,
                  metadata_path=args.metadata_path,
                  num_workers=4)
    print(f"Done. Index saved to {args.index_path}")


if __name__ == "__main__":
    main()
