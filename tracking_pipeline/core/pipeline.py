"""
Tracking + VLM classification pipeline.

Luồng (đúng theo sơ đồ):

    Ảnh ──► detection + tracking ──► object {class, id, đã phân loại?, class_vlm}
                                              │
                                   ┌──────────┴───────────┐
                                   │  đã phân loại chưa?   │
                                   └──────────┬───────────┘
                              chưa            │            rồi
                               ▼              │             ▼
                          VLM phân loại       │         dùng cache
                          update class_vlm ───┘         (kết thúc)

Điểm mấu chốt: mỗi track_id chỉ gọi VLM 1 LẦN. Sau khi có class_vlm,
các frame sau dùng lại cache → tiết kiệm rất nhiều compute.
"""

from typing import Dict, List, Optional

import numpy as np

from .types import TrackedObject, FrameResult
from ..detectors.base import BaseDetector
from ..trackers.bytetrack import ByteTrackWrapper
from ..vlm.base import BaseVLM


class TrackingVLMPipeline:
    def __init__(
        self,
        detector: BaseDetector,
        tracker: ByteTrackWrapper,
        vlm: BaseVLM,
        crop_padding: float = 0.05,
        min_vlm_score: float = 0.0,
        reclassify_after: Optional[int] = None,
    ):
        """
        Args:
            crop_padding: nới rộng bbox khi crop (tỉ lệ), giúp VLM có context.
            min_vlm_score: nếu VLM score < ngưỡng, KHÔNG mark classified
                           (sẽ thử lại frame sau).
            reclassify_after: nếu set N, cho phép phân loại lại sau N frame
                              (None = chỉ phân loại 1 lần duy nhất).
        """
        self.detector = detector
        self.tracker = tracker
        self.vlm = vlm
        self.crop_padding = crop_padding
        self.min_vlm_score = min_vlm_score
        self.reclassify_after = reclassify_after

        # cache trạng thái object theo track_id (đây là "trí nhớ" của pipeline)
        self.objects: Dict[int, TrackedObject] = {}

    def _crop(self, image: np.ndarray, bbox) -> np.ndarray:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        pad_x = (x2 - x1) * self.crop_padding
        pad_y = (y2 - y1) * self.crop_padding
        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(w, int(x2 + pad_x))
        y2 = min(h, int(y2 + pad_y))
        return image[y1:y2, x1:x2]

    def _needs_classification(self, obj: TrackedObject, frame_idx: int) -> bool:
        """Node quyết định 'đã phân loại chưa?'."""
        if not obj.classified:
            return True
        if self.reclassify_after is not None:
            # cho phép refresh định kỳ
            if frame_idx - obj.last_seen_frame_classified >= self.reclassify_after:
                return True
        return False

    def process_frame(self, image: np.ndarray, frame_idx: int) -> FrameResult:
        # 1. detection
        detections = self.detector.detect(image)

        # 2. tracking → gán id ổn định
        tracked = self.tracker.update(detections)

        result = FrameResult(frame_idx=frame_idx)

        for track_id, det in tracked:
            obj = self.objects.get(track_id)
            if obj is None:
                obj = TrackedObject(
                    track_id=track_id,
                    bbox=det.bbox,
                    det_class=det.cls,
                    det_score=det.score,
                )
                self.objects[track_id] = obj
            else:
                # cập nhật vị trí + class detector mới nhất
                obj.bbox = det.bbox
                obj.det_class = det.cls
                obj.det_score = det.score

            obj.last_seen_frame = frame_idx

            # 3. node quyết định: đã phân loại chưa?
            if self._needs_classification(obj, frame_idx):
                crop = self._crop(image, det.bbox)
                cls_vlm, score = self.vlm.classify(crop)
                result.vlm_calls += 1

                if score >= self.min_vlm_score:
                    obj.class_vlm = cls_vlm
                    obj.vlm_score = score
                    obj.classified = True
                    obj.last_seen_frame_classified = frame_idx
            # else: 'rồi' → dùng cache, không gọi VLM

            result.objects.append(obj)

        return result

    def reset(self):
        """Reset state (dùng khi bắt đầu video/sequence mới)."""
        self.objects.clear()
