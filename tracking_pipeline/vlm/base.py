"""Abstract VLM classifier interface.

VLM ở đây làm nhiệm vụ: nhận 1 crop ảnh của object → trả về 1 nhãn
trong 10 class quân sự (+ score). Chỉ được gọi 1 lần / track id (cache).
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BaseVLM(ABC):
    @abstractmethod
    def classify(self, crop: np.ndarray) -> Tuple[str, float]:
        """
        Args:
            crop: ảnh BGR của object (đã cắt theo bbox).
        Returns:
            (class_name, score) — class_name nằm trong MILITARY_CLASSES.
        """
        raise NotImplementedError
