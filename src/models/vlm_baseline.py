"""LLaVA baseline inference — no RAG."""
import torch
import logging
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor

logger = logging.getLogger(__name__)


class LLaVABaseline:
    """Plain LLaVA-1.5 inference without retrieval augmentation."""

    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf",
                 device: str = "cuda", dtype: str = "float16"):
        logger.info(f"Loading {model_name}...")
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def generate(self, image: Image.Image, question: str,
                 max_new_tokens: int = 128) -> str:
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        inputs = self.processor(
            text=prompt, images=image,
            return_tensors="pt"
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        # Strip the input tokens
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    def answer_yes_no(self, image: Image.Image, question: str) -> str:
        """Constrained yes/no for POPE evaluation."""
        answer = self.generate(image, question, max_new_tokens=10)
        answer_lower = answer.lower().strip()
        if answer_lower.startswith("yes"):
            return "yes"
        elif answer_lower.startswith("no"):
            return "no"
        return answer_lower
