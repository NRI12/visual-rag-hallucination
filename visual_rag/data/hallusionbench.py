"""HallusionBench dataset loader.
Source: https://github.com/tianyi-lab/HallusionBench
"""
import json
import os
import logging
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def normalize_gt(answer) -> str:
    """Normalize gt_answer to 'yes' or 'no' string."""
    if isinstance(answer, (int, float)):
        return "yes" if int(answer) == 1 else "no"
    s = str(answer).lower().strip().rstrip(".,!?")
    if s in ("yes", "true", "1", "correct"):
        return "yes"
    if s in ("no", "false", "0", "incorrect"):
        return "no"
    return s


class HallusionBenchDataset(Dataset):
    """
    HallusionBench: tests VLM robustness to visual hallucinations.
    gt_answer is normalized to 'yes'/'no' string.
    """

    def __init__(self, data_dir: str, max_samples: int = None):
        # Try multiple possible JSON filenames
        json_path = None
        for fname in ["HallusionBench.json", "hallusionbench.json",
                      "HallusionBench_anno.json", "annotation.json"]:
            candidate = os.path.join(data_dir, fname)
            if os.path.exists(candidate):
                json_path = candidate
                break

        if json_path is None:
            files = os.listdir(data_dir) if os.path.exists(data_dir) else []
            raise FileNotFoundError(
                f"HallusionBench JSON not found in {data_dir}. "
                f"Files present: {files[:15]}"
            )

        logger.info(f"Loading HallusionBench from {json_path}")
        with open(json_path) as f:
            raw = json.load(f)

        # Handle both list and dict-of-lists formats
        if isinstance(raw, dict):
            items = []
            for v in raw.values():
                if isinstance(v, list):
                    items.extend(v)
        else:
            items = raw

        self.data = []
        self.data_dir = data_dir

        for idx, item in enumerate(items):
            gt_raw = (item.get("gt_answer") or item.get("answer")
                      or item.get("label") or "")
            gt = normalize_gt(gt_raw)

            # HallusionBench actual keys: "filename" and "visual_input"
            # visual_input: "figure" = has image, "text" = text-only
            img_src = (item.get("filename") or item.get("image_src")
                       or item.get("image_path") or item.get("image") or "")
            img_src = img_src.lstrip("/") if img_src else ""
            visual_input = item.get("visual_input", "figure")
            is_visual = visual_input == "figure" and bool(img_src)

            self.data.append({
                "question":      item.get("question", item.get("text", "")),
                "gt_answer":     gt,
                "gt_answer_raw": gt_raw,
                "image_src":     img_src,
                "visual_input":  visual_input,
                "is_visual":     is_visual,
                "category":      item.get("category", ""),
                "subcategory":   item.get("sub_category", ""),
                "set_id":        item.get("set_id", ""),
                "item_id":       item.get("question_id", idx),
            })

        if max_samples:
            self.data = self.data[:max_samples]

        with_img = sum(1 for d in self.data if d["is_visual"])
        text_only = len(self.data) - with_img
        logger.info(f"Loaded {len(self.data)} HallusionBench samples "
                    f"({with_img} visual, {text_only} text-only).")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx].copy()
        img_src = item.get("image_src", "")

        item["image"] = None
        if img_src and item.get("is_visual"):
            # filename in HallusionBench is like "VD_E1_1.png" or "VG_E1_1.png"
            # images stored under data/hallusionbench/images/<set_id>/<filename>
            set_id = item.get("set_id", "")
            candidates = [
                os.path.join(self.data_dir, "images", set_id, img_src),
                os.path.join(self.data_dir, "images", img_src),
                os.path.join(self.data_dir, set_id, img_src),
                os.path.join(self.data_dir, img_src),
                os.path.join(self.data_dir, os.path.basename(img_src)),
            ]
            for path in candidates:
                if os.path.exists(path):
                    try:
                        item["image"] = Image.open(path).convert("RGB")
                    except Exception as e:
                        logger.warning(f"Cannot open {path}: {e}")
                    break
            else:
                logger.debug(f"Image not found: {img_src} (set_id={set_id})")

        return item
