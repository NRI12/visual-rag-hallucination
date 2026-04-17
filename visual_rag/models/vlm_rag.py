"""LLaVA + Visual-RAG inference — retrieval-augmented generation."""
import torch
import logging
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
from visual_rag.retrieval import VisualRetriever

logger = logging.getLogger(__name__)

# Additive framing: facts help the model, not restrict it
SYSTEM_PROMPT = (
    "You are a helpful visual assistant. "
    "Use the image and any provided visual context to answer accurately."
)


class LLaVAWithRAG:
    """LLaVA-1.5 augmented with scene-graph retrieval context."""

    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf",
                 retriever: VisualRetriever = None,
                 device: str = "cuda", dtype: str = "float16"):
        logger.info(f"Loading {model_name} with RAG...")
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.retriever = retriever
        self.device = device

    @torch.no_grad()
    def generate(self, image: Image.Image, question: str,
                 max_new_tokens: int = 128) -> str:
        # Augment question with retrieved context
        if self.retriever is not None:
            augmented_question, retrieved_facts = self.retriever.augment_prompt(
                question, image, system_prefix=SYSTEM_PROMPT
            )
        else:
            augmented_question = question
            retrieved_facts = []

        prompt = f"USER: <image>\n{augmented_question}\nASSISTANT:"
        inputs = self.processor(
            text=prompt, images=image,
            return_tensors="pt"
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        answer = self.processor.decode(generated, skip_special_tokens=True).strip()
        return answer, retrieved_facts

    def answer_yes_no(self, image: Image.Image, question: str) -> str:
        answer, _ = self.generate(image, question, max_new_tokens=10)
        answer_lower = answer.lower().strip()
        if answer_lower.startswith("yes"):
            return "yes"
        elif answer_lower.startswith("no"):
            return "no"
        return answer_lower
