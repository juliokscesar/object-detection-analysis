from abc import ABC, abstractmethod
from typing import List

class BaseAnalysisTask(ABC):
    def __init__(self):
        self._ctx_imgs = None
        self._ctx_detections = None
        self._ctx_masks = None
        self._ctx_data_cls = None
        self._require_masks = False
        self._require_detections = True

    @abstractmethod
    def run(self):
        return None
    
    @property
    def require_detections(self):
        return self._require_detections
    
    @property
    def require_masks(self):
        return self._require_masks

    def set_ctx(self, imgs: List[str], detections, masks, data_classes: List[str] = None):
        self.set_images(imgs)
        self.set_detections(detections)
        self.set_masks(masks)
        self.set_data_classes(data_classes)

    def set_images(self, imgs: List[str]):
        self._ctx_imgs = imgs
    
    def set_detections(self, detections):
        self._ctx_detections = detections

    def set_masks(self, masks):
        self._ctx_masks = masks

    def set_data_classes(self, data_classes: List[str]):
        self._ctx_data_cls = data_classes
