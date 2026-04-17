"""Main evaluation pipeline for baseline vs RAG comparison."""
import json
import os
import logging
from tqdm import tqdm
from typing import Dict, List

from visual_rag.evaluation.metrics import compute_pope_metrics, compute_chair

logger = logging.getLogger(__name__)


class POPEEvaluator:
    def __init__(self, model, output_dir: str = "results"):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(self, dataset, split_name: str = "adversarial",
            model_tag: str = "baseline") -> Dict:
        predictions, labels = [], []
        records = []

        for item in tqdm(dataset, desc=f"[{model_tag}] POPE {split_name}"):
            image = item["image"]
            question = item["question"]
            label = item["label"]

            if hasattr(self.model, "answer_yes_no"):
                pred = self.model.answer_yes_no(image, question)
            else:
                pred = self.model.generate(image, question, max_new_tokens=10)
                pred = pred.lower().strip()

            predictions.append(pred)
            labels.append(label)
            records.append({
                "image_id": item["image_id"],
                "question": question,
                "label": label,
                "prediction": pred,
                "correct": pred == label,
            })

        metrics = compute_pope_metrics(predictions, labels)
        metrics["split"] = split_name
        metrics["model"] = model_tag
        metrics["n_samples"] = len(predictions)

        # Save predictions
        out_file = os.path.join(
            self.output_dir, f"pope_{split_name}_{model_tag}.json"
        )
        with open(out_file, "w") as f:
            json.dump({"metrics": metrics, "predictions": records}, f, indent=2)
        logger.info(f"Results saved → {out_file}")
        logger.info(f"[{model_tag}] {split_name}: Acc={metrics['accuracy']}% F1={metrics['f1']}%")
        return metrics


class HallusionEvaluator:
    def __init__(self, model, output_dir: str = "results"):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(self, dataset, model_tag: str = "baseline") -> Dict:
        correct = 0
        records = []

        for item in tqdm(dataset, desc=f"[{model_tag}] HallusionBench"):
            image = item.get("image")
            if image is None:
                continue
            question = item["question"]
            gt = item["gt_answer"]

            if hasattr(self.model, "generate"):
                if hasattr(self.model, "retriever"):
                    answer, _ = self.model.generate(image, question, max_new_tokens=50)
                else:
                    answer = self.model.generate(image, question, max_new_tokens=50)
            else:
                answer = ""

            pred = answer.lower().strip()
            is_correct = gt in pred or pred.startswith(gt)
            if is_correct:
                correct += 1
            records.append({
                "item_id": item["item_id"],
                "question": question,
                "gt_answer": gt,
                "prediction": pred,
                "correct": is_correct,
                "category": item.get("category", ""),
            })

        accuracy = correct / len(records) if records else 0
        metrics = {
            "model": model_tag,
            "accuracy": round(accuracy * 100, 2),
            "n_samples": len(records),
            "n_correct": correct,
        }

        out_file = os.path.join(
            self.output_dir, f"hallusionbench_{model_tag}.json"
        )
        with open(out_file, "w") as f:
            json.dump({"metrics": metrics, "predictions": records}, f, indent=2)
        logger.info(f"[{model_tag}] HallusionBench: Acc={metrics['accuracy']}%")
        return metrics
