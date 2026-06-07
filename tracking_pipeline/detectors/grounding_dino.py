"""
GroundingDINO detector adapter.

Hỗ trợ load model fine-tuned của bạn. Mặc định dùng HuggingFace
`transformers` API (AutoProcessor + AutoModelForZeroShotObjectDetection)
vì nó ổn định và dễ load checkpoint fine-tuned.

Nếu bạn fine-tune bằng repo gốc IDEA-Research/GroundingDINO, hãy dùng
adapter này như tham chiếu rồi thay phần load model cho phù hợp.
"""

from typing import List, Optional

import numpy as np

from .base import BaseDetector
from ..core.types import Detection
from ..core.classes import MILITARY_CLASSES


class GroundingDINODetector(BaseDetector):
    def __init__(
        self,
        model_path: str,
        classes: Optional[List[str]] = None,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
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

        # GroundingDINO prompt: các class nối bằng " . "
        self.text_prompt = ". ".join(c.lower() for c in self.classes) + " ."

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(model_path)
            .to(device)
            .eval()
        )

        # map lowercase → tên class chuẩn để khôi phục đúng tên
        self._lower_to_canonical = {c.lower(): c for c in self.classes}

    def _canonicalize(self, phrase: str) -> Optional[str]:
        """Khớp phrase model trả về với 1 trong các class chuẩn."""
        phrase = phrase.strip().lower()
        if phrase in self._lower_to_canonical:
            return self._lower_to_canonical[phrase]
        # fallback: substring match (GroundingDINO đôi khi trả phrase 1 phần)
        for low, canon in self._lower_to_canonical.items():
            if phrase and (phrase in low or low in phrase):
                return canon
        return None

    def detect(self, image: np.ndarray) -> List[Detection]:
        from PIL import Image

        # BGR (cv2) → RGB → PIL
        rgb = image[:, :, ::-1]
        pil = Image.fromarray(rgb)

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
            target_sizes=[pil.size[::-1]],  # (h, w)
        )[0]

        detections: List[Detection] = []
        for box, score, label in zip(
            results["boxes"], results["scores"], results["labels"]
        ):
            canon = self._canonicalize(str(label))
            if canon is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            detections.append(
                Detection(bbox=(x1, y1, x2, y2), cls=canon, score=float(score))
            )
        return detections
