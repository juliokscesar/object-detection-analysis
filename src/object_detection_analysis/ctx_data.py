from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class ContextDetectionBoxData:
    object_classes: List[str]
    all_boxes: np.ndarray
    class_boxes: dict[str, np.ndarray]

@dataclass
class ContextDetectionMaskData:
    object_classes: List[str]
    all_masks: np.ndarray
    class_masks: dict[str, np.ndarray]
