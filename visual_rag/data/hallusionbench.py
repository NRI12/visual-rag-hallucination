"""HallusionBench dataset loader.
Source: https://github.com/tianyi-lab/HallusionBench
Contains 1,129 visual questions testing illusions, spatial reasoning, etc.
"""
import json
import os
import logging
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class HallusionBenchDataset(Dataset):
    """
    HallusionBench: tests VLM robustness to visual hallucinations.
    Each sample: image (or None for text-only), question, gt_answer.
    """

    def __init__(self, data_dir: str, max_samples: int = None):
        json_path = os.path.join(data_dir, "HallusionBench.json")
        with open(json_path) as f:
            raw = json.load(f)

        self.data = []
        self.data_dir = data_dir
        for item in raw:
            # Some items have no image (language-only hallucinations)
            self.data.append({
                "question": item.get("question", ""),
                "gt_answer": str(item.get("gt_answer", "")).lower(),
                "gt_answer_details": item.get("gt_answer_details", ""),
                "image_src": item.get("image_src", None),
                "category": item.get("category", ""),
                "subcategory": item.get("sub_category", ""),
                "item_id": item.get("index", len(self.data)),
            })

        if max_samples:
            self.data = self.data[:max_samples]
        logger.info(f"Loaded {len(self.data)} HallusionBench samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx].copy()
        img_src = item.get("image_src")
        if img_src:
            img_path = os.path.join(self.data_dir, img_src)
            try:
                item["image"] = Image.open(img_path).convert("RGB")
            except Exception:
                item["image"] = None
        else:
            item["image"] = None
        return item
