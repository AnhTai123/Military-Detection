"""
YOLO detector adapter (ultralytics).

Dùng model YOLO (v8/v9/v10/v11) bất kỳ. Class name từ model.names phải
khớp với MILITARY_CLASSES — nếu bạn train YOLO trên cùng dataset thì tự động khớp.
"""

from typing import List, Optional

import numpy as np

from .base import BaseDetector
from ..core.types import Detection
from ..core.classes import MILITARY_CLASSES


class YOLODetector(BaseDetector):
    def __init__(
        self,
        model_path: str,
        classes: Optional[List[str]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        imgsz: int = 640,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.model.to(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.imgsz = imgsz

        # tên class hợp lệ
        self.valid_classes = set(classes or MILITARY_CLASSES)

        # map model.names → canonical (để xử lý nếu có khác biệt nhỏ)
        self._name_map = {}
        lower_valid = {c.lower(): c for c in self.valid_classes}
        for idx, name in self.model.names.items():
            lo = name.lower().strip()
            canon = lower_valid.get(lo) or name
            self._name_map[idx] = canon

    def detect(self, image: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )
        detections: List[Detection] = []
        for res in results:
            if res.boxes is None:
                continue
            for box in res.boxes:
                cls_idx = int(box.cls[0])
                cls_name = self._name_map.get(cls_idx, "")
                if cls_name not in self.valid_classes:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                score = float(box.conf[0])
                detections.append(
                    Detection(bbox=(x1, y1, x2, y2), cls=cls_name, score=score)
                )
        return detections
