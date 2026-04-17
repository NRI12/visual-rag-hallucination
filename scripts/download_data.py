"""
Download all required datasets for Visual-RAG pipeline.
Usage:
    python scripts/download_data.py --data_dir data
"""
import argparse
import os
import sys
import zipfile
import shutil
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded * 100 / total_size, 100)
        bar = int(pct / 2)
        print(f"\r  [{'=' * bar}{' ' * (50 - bar)}] {pct:.1f}%", end="", flush=True)


def download_file(url: str, dest: str, desc: str = ""):
    os.makedirs(os.path.dirname(dest) if os.path.dirname(dest) else ".", exist_ok=True)
    print(f"  Downloading {desc or os.path.basename(dest)}...")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
        print()  # newline after progress bar
        return True
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return False


def download_pope(data_dir: str) -> bool:
    pope_dir = os.path.join(data_dir, "pope")
    marker = os.path.join(pope_dir, "coco_pope_adversarial.json")
    if os.path.exists(marker):
        print("  POPE already downloaded, skipping.")
        return True

    os.makedirs(pope_dir, exist_ok=True)
    base = "https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco"
    splits = ["adversarial", "popular", "random"]
    for split in splits:
        url = f"{base}/coco_pope_{split}.json"
        dest = os.path.join(pope_dir, f"coco_pope_{split}.json")
        if not download_file(url, dest, f"POPE {split}"):
            return False
    print("  POPE OK ✓")
    return True


def download_hallusionbench(data_dir: str) -> bool:
    hb_dir = os.path.join(data_dir, "hallusionbench")
    marker = os.path.join(hb_dir, "HallusionBench.json")
    # Also check if images already present
    img_dir = os.path.join(hb_dir, "images")
    has_images = os.path.exists(img_dir) and len(os.listdir(img_dir)) > 0

    if os.path.exists(marker) and has_images:
        print("  HallusionBench already downloaded, skipping.")
        return True

    os.makedirs(hb_dir, exist_ok=True)
    zip_path = "/tmp/hallusionbench.zip"
    url = "https://github.com/tianyi-lab/HallusionBench/archive/refs/heads/main.zip"

    if not download_file(url, zip_path, "HallusionBench (~300MB)"):
        return False

    print("  Extracting HallusionBench...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("/tmp/")

    src = "/tmp/HallusionBench-main"
    # Copy ALL contents including images/ subdirectory
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(hb_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

    os.remove(zip_path)
    # Check images were extracted
    n_images = sum(
        len(files) for _, _, files in os.walk(img_dir)
    ) if os.path.exists(img_dir) else 0
    print(f"  HallusionBench OK ✓ ({n_images} images)")
    return True


def download_coco_val2014(data_dir: str) -> bool:
    coco_dir = os.path.join(data_dir, "coco", "val2014")
    if os.path.exists(coco_dir) and len(os.listdir(coco_dir)) > 1000:
        print("  COCO val2014 already downloaded, skipping.")
        return True

    os.makedirs(os.path.join(data_dir, "coco"), exist_ok=True)
    zip_path = os.path.join(data_dir, "coco", "val2014.zip")
    url = "http://images.cocodataset.org/zips/val2014.zip"

    print("  Downloading COCO val2014 (~7GB)...")
    if not download_file(url, zip_path, "COCO val2014"):
        return False

    print("  Extracting COCO val2014 (this takes a few minutes)...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(os.path.join(data_dir, "coco"))

    os.remove(zip_path)
    print("  COCO val2014 OK ✓")
    return True


def download_visual_genome(data_dir: str) -> bool:
    vg_dir = os.path.join(data_dir, "visual_genome")
    rel_file = os.path.join(vg_dir, "relationships.json")
    attr_file = os.path.join(vg_dir, "attributes.json")

    if os.path.exists(rel_file) and os.path.exists(attr_file):
        print("  Visual Genome already downloaded, skipping.")
        return True

    os.makedirs(vg_dir, exist_ok=True)
    base = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset"
    files = ["relationships.json.zip", "attributes.json.zip"]

    for fname in files:
        zip_path = os.path.join(vg_dir, fname)
        if not download_file(f"{base}/{fname}", zip_path, fname):
            return False
        print(f"  Extracting {fname}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(vg_dir)
        os.remove(zip_path)

    print("  Visual Genome OK ✓")
    return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--skip_coco", action="store_true",
                   help="Skip COCO download (needed for POPE eval)")
    p.add_argument("--skip_vg", action="store_true",
                   help="Skip Visual Genome (needed for index build)")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir
    print(f"\n{'=' * 50}")
    print(f" Downloading datasets to: {os.path.abspath(data_dir)}")
    print(f"{'=' * 50}\n")

    steps = [
        ("POPE",            lambda: download_pope(data_dir)),
        ("HallusionBench",  lambda: download_hallusionbench(data_dir)),
    ]
    if not args.skip_coco:
        steps.append(("COCO val2014", lambda: download_coco_val2014(data_dir)))
    if not args.skip_vg:
        steps.append(("Visual Genome", lambda: download_visual_genome(data_dir)))

    failed = []
    for name, fn in steps:
        print(f"[{name}]")
        ok = fn()
        if not ok:
            failed.append(name)
        print()

    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print("All datasets ready ✓")


if __name__ == "__main__":
    main()
