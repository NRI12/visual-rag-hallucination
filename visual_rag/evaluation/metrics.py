"""Evaluation metrics: Accuracy, F1, CHAIR_S, CHAIR_I."""
import json
import logging
from collections import defaultdict
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


def compute_pope_metrics(predictions: List[str], labels: List[str]) -> Dict:
    """Compute accuracy, precision, recall, F1 for POPE yes/no task."""
    assert len(predictions) == len(labels)
    tp = fp = tn = fn = 0
    for pred, label in zip(predictions, labels):
        pred = pred.strip().lower()
        label = label.strip().lower()
        if pred == "yes" and label == "yes":
            tp += 1
        elif pred == "yes" and label == "no":
            fp += 1
        elif pred == "no" and label == "no":
            tn += 1
        elif pred == "no" and label == "yes":
            fn += 1

    accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    yes_rate = (tp + fp) / (tp + fp + tn + fn + 1e-9)

    return {
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "yes_rate": round(yes_rate * 100, 2),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def compute_chair(predictions: List[str], gt_objects: List[List[str]],
                  coco_objects: List[str] = None) -> Dict:
    """
    CHAIR_S (sentence-level) and CHAIR_I (instance-level) hallucination rates.
    predictions: list of caption strings
    gt_objects:  list of ground-truth object sets per image
    coco_objects: flat list of all COCO object categories (80 classes)
    """
    if coco_objects is None:
        coco_objects = _default_coco_objects()

    hallucinated_sentences = 0
    total_sentences = len(predictions)
    total_hal_mentions = 0
    total_mentions = 0

    for pred, gt in zip(predictions, gt_objects):
        gt_set = set(o.lower() for o in gt)
        mentioned = [obj for obj in coco_objects if obj in pred.lower()]
        total_mentions += len(mentioned)
        hal_mentions = [obj for obj in mentioned if obj not in gt_set]
        total_hal_mentions += len(hal_mentions)
        if hal_mentions:
            hallucinated_sentences += 1

    chair_s = hallucinated_sentences / (total_sentences + 1e-9)
    chair_i = total_hal_mentions / (total_mentions + 1e-9)

    return {
        "CHAIR_S": round(chair_s * 100, 2),
        "CHAIR_I": round(chair_i * 100, 2),
        "hallucinated_sentences": hallucinated_sentences,
        "total_sentences": total_sentences,
    }


def _default_coco_objects() -> List[str]:
    return [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
    ]
