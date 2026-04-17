"""CLIP-based encoder for image and text embeddings."""
import torch
import numpy as np
import open_clip
from PIL import Image
from typing import Union, List


class CLIPEncoder:
    """Encodes images and text using OpenCLIP for retrieval."""

    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai",
                 device: str = "cuda"):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(device).eval()
        # infer embedding dim from model
        self.embed_dim = self.model.visual.output_dim

    @torch.no_grad()
    def encode_image(self, image: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        if isinstance(image, Image.Image):
            image = [image]
        tensors = torch.stack([self.preprocess(img) for img in image]).to(self.device)
        feats = self.model.encode_image(tensors)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().numpy()

    @torch.no_grad()
    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        tokens = self.tokenizer(texts).to(self.device)
        feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().numpy()

    def encode_query(self, image: Image.Image, question: str,
                     image_weight: float = 0.7) -> np.ndarray:
        """Fuse image + text embeddings for query."""
        img_emb = self.encode_image(image)
        txt_emb = self.encode_text(question)
        fused = image_weight * img_emb + (1 - image_weight) * txt_emb
        fused = fused / np.linalg.norm(fused, axis=-1, keepdims=True)
        return fused
