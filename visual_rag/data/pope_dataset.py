"""POPE benchmark dataset loader.
Source: https://github.com/AoiDragon/POPE
Splits: adversarial, popular, random (each ~3000 questions)
"""
import json
import os
import logging
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

POPE_SPLITS = ["adversarial", "popular", "random"]


class POPEDataset(Dataset):
    """
    POPE yes/no hallucination benchmark.
    Each sample: image, question, label (yes/no).
    """

    def __init__(self, pope_dir: str, coco_dir: str,
                 split: str = "adversarial", max_samples: int = None):
        assert split in POPE_SPLITS, f"split must be one of {POPE_SPLITS}"
        self.coco_dir = coco_dir
        annotation_file = os.path.join(
            pope_dir, f"coco_pope_{split}.json"
        )
        with open(annotation_file) as f:
            data = [json.loads(line) for line in f]
        if max_samples:
            data = data[:max_samples]
        self.data = data
        logger.info(f"Loaded {len(self.data)} POPE {split} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.coco_dir, item["image"])
        image = Image.open(image_path).convert("RGB")
        return {
            "image": image,
            "question": item["text"],
            "label": item["label"].lower(),   # "yes" or "no"
            "image_id": item["image"],
        }


def load_pope_all_splits(pope_dir: str, coco_dir: str,
                         max_samples: int = None):
    return {
        split: POPEDataset(pope_dir, coco_dir, split, max_samples)
        for split in POPE_SPLITS
    }
