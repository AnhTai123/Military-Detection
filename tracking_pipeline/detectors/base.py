"""Abstract detector interface."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ..core.types import Detection


class BaseDetector(ABC):
    """Mọi detector phải trả về list[Detection] từ 1 ảnh BGR (numpy)."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Args:
            image: ảnh BGR, shape (H, W, 3), dtype uint8.
        Returns:
            list các Detection (bbox xyxy pixel, cls name, score).
        """
        raise NotImplementedError
