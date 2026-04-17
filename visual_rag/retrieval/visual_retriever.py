"""Main Visual-RAG retrieval pipeline."""
import re
import logging
from PIL import Image
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


def extract_target_object(question: str) -> Optional[str]:
    """
    Extract the target object from POPE-style yes/no questions.
    e.g. "Is there a dog in the image?" → "dog"
         "Are there any people in the scene?" → "people"
    """
    q = question.lower().strip()
    patterns = [
        r"is there (?:a|an|any) (.+?)(?:\s+in|\s+on|\s+at|\s+visible|\?|$)",
        r"are there (?:any )? (.+?)(?:\s+in|\s+on|\s+at|\s+visible|\?|$)",
        r"do you see (?:a|an|any) (.+?)(?:\s+in|\?|$)",
        r"can you see (?:a|an|any) (.+?)(?:\s+in|\?|$)",
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            obj = m.group(1).strip().rstrip("?.,")
            return obj
    return None


class VisualRetriever:
    """
    Given an image + question, retrieves relevant scene graph facts.
    Key insight: filter retrieved facts to those mentioning the target
    object in the question — prevents adding unrelated noise.
    """

    def __init__(self, encoder, indexer, top_k: int = 10,
                 score_threshold: float = 0.28, image_weight: float = 0.6):
        self.encoder = encoder
        self.indexer = indexer
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.image_weight = image_weight

    def retrieve(self, image: Image.Image, question: str) -> List[Tuple[str, float]]:
        # Use text-only query for better concept matching
        txt_emb  = self.encoder.encode_text(question)
        img_emb  = self.encoder.encode_image(image)
        import numpy as np
        query_emb = (self.image_weight * img_emb +
                     (1 - self.image_weight) * txt_emb)
        query_emb = query_emb / (np.linalg.norm(query_emb, axis=-1, keepdims=True) + 1e-9)
        facts = self.indexer.search(query_emb, self.top_k, self.score_threshold)

        # Filter: keep only facts mentioning the target object (if extractable)
        target = extract_target_object(question)
        if target and facts:
            filtered = [(f, s) for f, s in facts
                        if any(w in f.lower() for w in target.split())]
            # Fall back to top-3 unfiltered if filter removes everything
            facts = filtered if filtered else facts[:3]

        return facts[:5]  # cap at 5

    def format_context(self, facts: List[Tuple[str, float]]) -> str:
        if not facts:
            return ""
        seen, lines = set(), []
        for fact, score in facts:
            if fact not in seen:
                lines.append(f"- {fact}")
                seen.add(fact)
        return ("Background visual knowledge (from knowledge base, "
                "NOT necessarily about this image):\n" + "\n".join(lines))

    def augment_prompt(self, question: str, image: Image.Image,
                       system_prefix: Optional[str] = None) -> Tuple[str, List]:
        facts = self.retrieve(image, question)
        context = self.format_context(facts)

        parts = []
        if system_prefix:
            parts.append(system_prefix)
        if context:
            parts.append(context)
        parts.append(question)

        prompt = "\n\n".join(parts)
        return prompt, facts
