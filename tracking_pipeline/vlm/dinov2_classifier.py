"""
DINOv2 + linear head classifier — GIẢI PHÁP CHÍNH cho fine-grained.

DINOv2 (Meta) cho feature ảnh cực mạnh, đặc biệt tốt khi phân biệt các vật
nhìn gần giống nhau (T-90 vs M1 Abrams, F-22 vs F-35). Ta đóng băng DINOv2 và
chỉ train 1 lớp linear nhỏ (head) trên các crop GT của bạn -> rất nhẹ, rất nhanh,
chính xác hơn zero-shot CLIP/SigLIP nhiều.

Backbone facebook/dinov2-small ~21M params (nhẹ hơn LocateAnything-3B ~140 lần).

Cần train head trước bằng:  python -m tracking_pipeline.train_dino_head
Head lưu ra 1 file .pt, rồi dùng:  --vlm dinov2 --vlm-path dino_head.pt
"""

from typing import List, Optional, Tuple

import numpy as np

from .base import BaseVLM
from ..core.classes import MILITARY_CLASSES


class DINOv2Classifier(BaseVLM):
    def __init__(
        self,
        head_path: str,
        device: str = "cuda",
    ):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = device

        # nạp head đã train (chứa weight linear + danh sách class + tên backbone)
        ckpt = torch.load(head_path, map_location=device)
        self.classes: List[str] = ckpt["classes"]
        dino_model: str = ckpt["dino_model"]
        feat_dim: int = ckpt["feat_dim"]

        self.processor = AutoImageProcessor.from_pretrained(dino_model)
        self.backbone = AutoModel.from_pretrained(dino_model).to(device).eval()

        self.head = torch.nn.Linear(feat_dim, len(self.classes)).to(device)
        self.head.load_state_dict(ckpt["head_state"])
        self.head.eval()

    @staticmethod
    def extract_feature(backbone, processor, pil, device, torch):
        """Trích feature DINOv2 (CLS pooled) từ 1 ảnh PIL."""
        inputs = processor(images=pil, return_tensors="pt").to(device)
        with torch.no_grad():
            out = backbone(**inputs)
            # pooler_output = CLS token đã pool; fallback mean nếu không có
            if getattr(out, "pooler_output", None) is not None:
                feat = out.pooler_output
            else:
                feat = out.last_hidden_state[:, 0]
        return feat  # (1, dim)

    def classify(self, crop: np.ndarray) -> Tuple[str, float]:
        from PIL import Image

        if crop.size == 0:
            return self.classes[0], 0.0

        pil = Image.fromarray(crop[:, :, ::-1])
        feat = self.extract_feature(
            self.backbone, self.processor, pil, self.device, self.torch
        )
        with self.torch.no_grad():
            logits = self.head(feat)
            probs = logits.softmax(dim=-1)[0]
        score, idx = probs.max(dim=-1)
        return self.classes[int(idx)], float(score)
