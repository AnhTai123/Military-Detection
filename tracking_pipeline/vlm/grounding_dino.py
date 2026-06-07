"""
GroundingDINO VLM adapter — dùng để phân loại crop (chạy 1 lần / track id).

Cách hoạt động:
    Nhận 1 crop BGR của object → chạy GDino với prompt tất cả 10 class →
    lấy class có detection score cao nhất → trả về (class_name, score).

Tại sao GDino làm VLM tốt:
    - Bạn đã fine-tune → hiểu rõ 10 class quân sự này
    - Open-vocab: chỉ cần prompt text → score ứng với từng class
    - Chính xác hơn YOLO cho crop nhỏ/khó (do text-visual matching)
    - Chạy 1 lần / track id nên không cần lo về tốc độ
"""

from typing import List, Optional, Tuple

import numpy as np

from .base import BaseVLM
from ..core.classes import MILITARY_CLASSES


class GroundingDINOVLM(BaseVLM):
    def __init__(
        self,
        model_path: str,
        classes: Optional[List[str]] = None,
        box_threshold: float = 0.1,   # thấp hơn khi dùng làm VLM (crop đã tight)
        text_threshold: float = 0.1,
        device: str = "cuda",
    ):
        import torch
        from transformers import (
            AutoProcessor,
            AutoModelForZeroShotObjectDetection,
        )

        self.torch = torch
        self.device = device
        self.classes = classes or MILITARY_CLASSES
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # prompt gồm tất cả 10 class
        self.text_prompt = ". ".join(c.lower() for c in self.classes) + " ."

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(model_path)
            .to(device)
            .eval()
        )
        self._lower_to_canonical = {c.lower(): c for c in self.classes}

    def _canonicalize(self, phrase: str) -> Optional[str]:
        phrase = phrase.strip().lower()
        if phrase in self._lower_to_canonical:
            return self._lower_to_canonical[phrase]
        # substring fallback
        for low, canon in self._lower_to_canonical.items():
            if phrase and (phrase in low or low in phrase):
                return canon
        return None

    def classify(self, crop: np.ndarray) -> Tuple[str, float]:
        """
        Chạy GDino trên crop → class có score cao nhất.
        Nếu không detect được gì, trả về class đầu tiên với score 0.
        """
        from PIL import Image

        if crop.size == 0:
            return self.classes[0], 0.0

        pil = Image.fromarray(crop[:, :, ::-1])

        inputs = self.processor(
            images=pil, text=self.text_prompt, return_tensors="pt"
        ).to(self.device)

        with self.torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[pil.size[::-1]],
        )[0]

        if len(results["scores"]) == 0:
            return self.classes[0], 0.0

        # lấy detection có score cao nhất → đó là class VLM chọn
        best_idx = results["scores"].argmax().item()
        best_score = float(results["scores"][best_idx])
        best_label = str(results["labels"][best_idx])
        canon = self._canonicalize(best_label) or self.classes[0]

        return canon, best_score
