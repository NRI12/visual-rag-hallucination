"""Main Visual-RAG retrieval pipeline."""
import logging
from PIL import Image
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class VisualRetriever:
    """
    Given an image + question, retrieves relevant scene graph facts
    from the FAISS index to ground VLM generation.
    """

    def __init__(self, encoder, indexer, top_k: int = 5,
                 score_threshold: float = 0.25, image_weight: float = 0.7):
        self.encoder = encoder
        self.indexer = indexer
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.image_weight = image_weight

    def retrieve(self, image: Image.Image, question: str) -> List[Tuple[str, float]]:
        query_emb = self.encoder.encode_query(image, question, self.image_weight)
        facts = self.indexer.search(query_emb, self.top_k, self.score_threshold)
        return facts

    def format_context(self, facts: List[Tuple[str, float]]) -> str:
        """Format retrieved facts as a natural language context string."""
        if not facts:
            return ""
        lines = ["Visual context from scene analysis:"]
        for fact, score in facts:
            lines.append(f"  - {fact}")
        return "\n".join(lines)

    def augment_prompt(self, question: str, image: Image.Image,
                       system_prefix: Optional[str] = None) -> str:
        """Build RAG-augmented prompt for the VLM."""
        facts = self.retrieve(image, question)
        context = self.format_context(facts)

        if context:
            prompt = f"{context}\n\nBased on the image and the above visual context, answer:\n{question}"
        else:
            prompt = question

        if system_prefix:
            prompt = f"{system_prefix}\n\n{prompt}"

        return prompt, facts
