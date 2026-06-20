"""
SigLIP2 zero-shot classifier — VLM nhẹ, mạnh hơn CLIP rõ rệt.

Cùng cơ chế CLIP (so khớp embedding ảnh với embedding text 10 class) nhưng
SigLIP2 (Google) cho độ chính xác cao hơn nhiều ở zero-shot. Không cần train.

Swap thẳng vào pipeline:  --vlm siglip
"""

from typing import List, Optional, Tuple

import numpy as np

from .base import BaseVLM
from ..core.classes import MILITARY_CLASSES


class SigLIPClassifier(BaseVLM):
    def __init__(
        self,
        classes: Optional[List[str]] = None,
        model_path: str = "google/siglip2-base-patch16-224",
        device: str = "cuda",
        prompt_template: str = "a photo of a {}, a military vehicle, aircraft or ship",
    ):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = device
        self.classes = classes or MILITARY_CLASSES

        self.model = AutoModel.from_pretrained(model_path).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

        # pre-compute text embeddings cho 10 class (1 lần)
        prompts = [prompt_template.format(c) for c in self.classes]
        text_inputs = self.processor(
            text=prompts, padding="max_length", return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            text_feat = self.model.get_text_features(**text_inputs)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        self.text_feat = text_feat  # (num_classes, dim)

        # logit scale/bias đặc trưng của SigLIP (để ra xác suất sigmoid hợp lý)
        self.logit_scale = self.model.logit_scale.exp() \
            if hasattr(self.model, "logit_scale") else torch.tensor(1.0, device=device)
        self.logit_bias = self.model.logit_bias \
            if hasattr(self.model, "logit_bias") else torch.tensor(0.0, device=device)

    def classify(self, crop: np.ndarray) -> Tuple[str, float]:
        from PIL import Image

        if crop.size == 0:
            return self.classes[0], 0.0

        pil = Image.fromarray(crop[:, :, ::-1])
        img_inputs = self.processor(images=pil, return_tensors="pt").to(self.device)

        with self.torch.no_grad():
            img_feat = self.model.get_image_features(**img_inputs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            logits = img_feat @ self.text_feat.T * self.logit_scale + self.logit_bias
            probs = self.torch.sigmoid(logits)[0]

        score, idx = probs.max(dim=-1)
        return self.classes[int(idx)], float(score)
