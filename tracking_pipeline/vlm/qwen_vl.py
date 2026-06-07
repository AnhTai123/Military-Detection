"""
Qwen2.5-VL classifier adapter (tùy chọn, nặng hơn CLIP).

Dùng khi muốn độ chính xác cao hơn. Mặc định pipeline dùng CLIP cho nhẹ.
Đây cũng là tham chiếu để sau này cắm LocateAnything-3B (cùng pattern):
chỉ cần thay phần load model + prompt.
"""

from typing import List, Optional, Tuple

import numpy as np

from .base import BaseVLM
from ..core.classes import MILITARY_CLASSES


class QwenVLClassifier(BaseVLM):
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        classes: Optional[List[str]] = None,
        device: str = "cuda",
    ):
        import torch
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            AutoProcessor,
        )

        self.torch = torch
        self.device = device
        self.classes = classes or MILITARY_CLASSES

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

        self._class_list_str = "\n".join(f"- {c}" for c in self.classes)
        self._lower_to_canonical = {c.lower(): c for c in self.classes}

    def _match_class(self, text: str) -> Tuple[str, float]:
        text_l = text.strip().lower()
        for low, canon in self._lower_to_canonical.items():
            if low in text_l:
                return canon, 1.0
        # fallback: token overlap
        best, best_score = self.classes[0], 0.0
        for low, canon in self._lower_to_canonical.items():
            overlap = len(set(low.split()) & set(text_l.split()))
            if overlap > best_score:
                best, best_score = canon, overlap
        return best, 0.5 if best_score else 0.0

    def classify(self, crop: np.ndarray) -> Tuple[str, float]:
        from PIL import Image

        if crop.size == 0:
            return self.classes[0], 0.0

        pil = Image.fromarray(crop[:, :, ::-1])
        prompt = (
            "Classify the military object in this image into exactly one of "
            f"these classes:\n{self._class_list_str}\n"
            "Answer with only the exact class name."
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[pil], return_tensors="pt"
        ).to(self.device)

        with self.torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=32)
        trimmed = gen[:, inputs.input_ids.shape[1]:]
        answer = self.processor.batch_decode(
            trimmed, skip_special_tokens=True
        )[0]
        return self._match_class(answer)
