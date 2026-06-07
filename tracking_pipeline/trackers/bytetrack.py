"""
ByteTrack wrapper (qua thư viện `supervision`).

Nhận list[Detection] mỗi frame, trả về list[(track_id, Detection)].
ByteTrack chỉ cần bbox + score, không cần GPU thêm — rất nhẹ.
"""

from typing import List, Optional, Tuple

import numpy as np

from ..core.types import Detection


class ByteTrackWrapper:
    def __init__(
        self,
        classes: List[str],
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
    ):
        import supervision as sv

        self.sv = sv
        self.classes = classes
        self._cls_to_idx = {c: i for i, c in enumerate(classes)}
        self._idx_to_cls = {i: c for i, c in enumerate(classes)}

        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

    def update(self, detections: List[Detection]) -> List[Tuple[int, Detection]]:
        if len(detections) == 0:
            empty = self.sv.Detections.empty()
            self.tracker.update_with_detections(empty)
            return []

        xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
        conf = np.array([d.score for d in detections], dtype=np.float32)
        class_id = np.array(
            [self._cls_to_idx.get(d.cls, 0) for d in detections], dtype=int
        )

        sv_det = self.sv.Detections(
            xyxy=xyxy, confidence=conf, class_id=class_id
        )
        tracked = self.tracker.update_with_detections(sv_det)

        out: List[Tuple[int, Detection]] = []
        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i])
            x1, y1, x2, y2 = [float(v) for v in tracked.xyxy[i]]
            cls = self._idx_to_cls.get(int(tracked.class_id[i]), self.classes[0])
            score = float(tracked.confidence[i]) if tracked.confidence is not None else 1.0
            out.append((tid, Detection(bbox=(x1, y1, x2, y2), cls=cls, score=score)))
        return out
