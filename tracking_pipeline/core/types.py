"""Core data structures for the tracking + VLM classification pipeline."""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class TrackedObject:
    """
    Một object được track qua các frame.

    Tương ứng node "object" trong sơ đồ:
        {
          class      : class do detector dự đoán
          id         : track id (ổn định qua các frame)
          classified : đã được VLM phân loại chưa? (đã phân loại?)
          class_vlm  : nhãn cuối cùng do VLM gán (update class_vlm)
        }
    """
    track_id: int
    bbox: Tuple[float, float, float, float]      # xyxy (pixel)
    det_class: str                                # class từ detector
    det_score: float
    classified: bool = False                      # đã phân loại bởi VLM chưa
    class_vlm: Optional[str] = None               # nhãn VLM (cache)
    vlm_score: Optional[float] = None
    last_seen_frame: int = -1
    last_seen_frame_classified: int = -(10 ** 9)  # frame gần nhất VLM phân loại

    @property
    def final_class(self) -> str:
        """Nhãn hiển thị: ưu tiên class_vlm nếu đã có, ngược lại dùng det_class."""
        return self.class_vlm if self.class_vlm is not None else self.det_class


@dataclass
class Detection:
    """Một detection thô từ detector (chưa track)."""
    bbox: Tuple[float, float, float, float]      # xyxy pixel
    cls: str
    score: float


@dataclass
class FrameResult:
    """Kết quả của 1 frame: danh sách objects + số lần gọi VLM trong frame."""
    frame_idx: int
    objects: List[TrackedObject] = field(default_factory=list)
    vlm_calls: int = 0
