"""Build and manage FAISS index from Visual Genome scene graphs."""
import json
import os
import logging
import numpy as np
import faiss
from tqdm import tqdm
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


def load_visual_genome_facts(vg_dir: str) -> List[Dict]:
    """
    Load Visual Genome relationships and attributes as flat fact strings.
    Returns list of dicts: {image_id, fact, objects, url}
    """
    rel_path = os.path.join(vg_dir, "relationships.json")
    attr_path = os.path.join(vg_dir, "attributes.json")

    facts = []

    if os.path.exists(rel_path):
        logger.info("Loading VG relationships...")
        with open(rel_path) as f:
            rels = json.load(f)
        for img in tqdm(rels, desc="Parsing relationships"):
            img_id = img["image_id"]
            for rel in img.get("relationships", []):
                subj = rel["subject"].get("name", "object")
                pred = rel.get("predicate", "related_to")
                obj_name = rel["object"].get("name", "object")
                fact = f"{subj} {pred} {obj_name}"
                facts.append({
                    "image_id": img_id,
                    "fact": fact,
                    "type": "relation",
                    "objects": [subj, obj_name]
                })

    if os.path.exists(attr_path):
        logger.info("Loading VG attributes...")
        with open(attr_path) as f:
            attrs = json.load(f)
        for img in tqdm(attrs, desc="Parsing attributes"):
            img_id = img["image_id"]
            for obj in img.get("attributes", []):
                name = obj.get("names", ["object"])[0]
                for attr in obj.get("attributes", []):
                    fact = f"{name} is {attr}"
                    facts.append({
                        "image_id": img_id,
                        "fact": fact,
                        "type": "attribute",
                        "objects": [name]
                    })

    logger.info(f"Loaded {len(facts)} facts from Visual Genome.")
    return facts


class SceneGraphIndexer:
    """Builds and queries a FAISS index over VG scene graph facts."""

    def __init__(self, encoder, dim: int = 768):
        self.encoder = encoder
        self.dim = dim
        self.index = None
        self.metadata: List[Dict] = []

    def build(self, facts: List[Dict], batch_size: int = 512,
              index_path: str = None, metadata_path: str = None):
        """Encode all facts and build FAISS flat index."""
        texts = [f["fact"] for f in facts]
        all_embeddings = []

        logger.info(f"Encoding {len(texts)} facts in batches of {batch_size}...")
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding facts"):
            batch = texts[i: i + batch_size]
            embs = self.encoder.encode_text(batch)
            all_embeddings.append(embs)

        embeddings = np.vstack(all_embeddings).astype("float32")

        self.index = faiss.IndexFlatIP(self.dim)  # Inner product = cosine (normalized)
        self.index.add(embeddings)
        self.metadata = facts

        logger.info(f"FAISS index built with {self.index.ntotal} vectors.")

        if index_path:
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            faiss.write_index(self.index, index_path)
            logger.info(f"Index saved to {index_path}")
        if metadata_path:
            with open(metadata_path, "w") as f:
                json.dump(facts, f)
            logger.info(f"Metadata saved to {metadata_path}")

    def load(self, index_path: str, metadata_path: str):
        self.index = faiss.read_index(index_path)
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        logger.info(f"Loaded index with {self.index.ntotal} vectors.")

    def search(self, query_emb: np.ndarray, top_k: int = 5,
               score_threshold: float = 0.2) -> List[Tuple[str, float]]:
        """Return top-k (fact, score) pairs above threshold."""
        if query_emb.ndim == 1:
            query_emb = query_emb[np.newaxis, :]
        query_emb = query_emb.astype("float32")

        scores, indices = self.index.search(query_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < score_threshold:
                continue
            results.append((self.metadata[idx]["fact"], float(score)))
        return results
