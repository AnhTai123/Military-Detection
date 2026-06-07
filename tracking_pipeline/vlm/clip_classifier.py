"""
CLIP zero-shot classifier — VLM nhẹ mặc định.

Phân loại 1 crop vào 10 class quân sự bằng cách so khớp embedding ảnh với
embedding text của từng class. Rất nhẹ (open_clip ViT-B/32 ~150MB) và nhanh,
phù hợp gọi trong tracking loop (dù thực tế chỉ gọi 1 lần / track id).
"""

from typing import List, Optional, Tuple

import numpy as np

from .base import BaseVLM
from ..core.classes import MILITARY_CLASSES


class CLIPClassifier(BaseVLM):
    def __init__(
        self,
        classes: Optional[List[str]] = None,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cuda",
        prompt_template: str = "a photo of a {}, military vehicle or aircraft",
    ):
        import torch
        import open_clip

        self.torch = torch
        self.device = device
        self.classes = classes or MILITARY_CLASSES

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # pre-compute text embeddings cho 10 class (làm 1 lần)
        prompts = [prompt_template.format(c) for c in self.classes]
        with torch.no_grad():
            tokens = self.tokenizer(prompts).to(device)
            text_feat = self.model.encode_text(tokens)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
        self.text_feat = text_feat  # (num_classes, dim)

    def classify(self, crop: np.ndarray) -> Tuple[str, float]:
        from PIL import Image

        if crop.size == 0:
            return self.classes[0], 0.0

        rgb = crop[:, :, ::-1]
        pil = Image.fromarray(rgb)
        img = self.preprocess(pil).unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            img_feat = self.model.encode_image(img)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            logits = (100.0 * img_feat @ self.text_feat.T).softmax(dim=-1)

        score, idx = logits[0].max(dim=-1)
        return self.classes[int(idx)], float(score)
