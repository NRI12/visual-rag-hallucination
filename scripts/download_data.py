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
    os.makedirs(hb_dir, exist_ok=True)

    # Step 1: JSON annotations from GitHub
    json_path = os.path.join(hb_dir, "HallusionBench.json")
    if not os.path.exists(json_path):
        zip_path = "/tmp/hallusionbench.zip"
        url = "https://github.com/tianyi-lab/HallusionBench/archive/refs/heads/main.zip"
        if not download_file(url, zip_path, "HallusionBench JSON (~5MB)"):
            return False
        print("  Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall("/tmp/")
        src = "/tmp/HallusionBench-main"
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(hb_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        os.remove(zip_path)
        print("  JSON OK ✓")

    # Step 2: Images via git clone (GitHub LFS)
    img_dir = os.path.join(hb_dir, "images")
    n_existing = sum(len(f) for _, _, f in os.walk(img_dir)) if os.path.exists(img_dir) else 0

    if n_existing < 100:
        print("  Downloading HallusionBench images via git clone (LFS)...")
        import subprocess
        clone_dir = "/tmp/HallusionBench_repo"
        try:
            # Install git-lfs silently
            subprocess.run(["git", "lfs", "install"], check=True,
                           capture_output=True)
            # Shallow clone to save time/space
            if not os.path.exists(clone_dir):
                subprocess.run([
                    "git", "clone", "--depth", "1",
                    "https://github.com/tianyi-lab/HallusionBench.git",
                    clone_dir
                ], check=True)
            # Copy images folder
            src_img = os.path.join(clone_dir, "images")
            if os.path.exists(src_img):
                shutil.copytree(src_img, img_dir, dirs_exist_ok=True)
                n_images = sum(len(f) for _, _, f in os.walk(img_dir))
                print(f"  Images OK ✓ ({n_images} images)")
            else:
                # LFS pointers only — pull actual files
                subprocess.run(["git", "lfs", "pull"], cwd=clone_dir, check=True)
                if os.path.exists(src_img):
                    shutil.copytree(src_img, img_dir, dirs_exist_ok=True)
                    n_images = sum(len(f) for _, _, f in os.walk(img_dir))
                    print(f"  Images OK ✓ ({n_images} images)")
                else:
                    print("  WARNING: No images/ folder in repo — using dummy images for eval")
        except Exception as e:
            print(f"  WARNING: git clone failed ({e}) — using dummy images for eval")
    else:
        print(f"  HallusionBench images already present ({n_existing} files), skipping.")

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
