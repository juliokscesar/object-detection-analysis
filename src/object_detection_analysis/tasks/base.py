from abc import ABC, abstractmethod
from typing import List
import numpy as np
import pandas as pd

from object_detection_analysis.ctx_data import ContextDetectionBoxData, ContextDetectionMaskData

class BaseAnalysisTask(ABC):
    def __init__(self):
        self._ctx_imgs: List[str] = None
        self._ctx_detections: dict[str, ContextDetectionBoxData] = None
        self._ctx_masks: dict[str, ContextDetectionMaskData] = None
        self._ctx_data_cls: List[str] = None
        self._require_masks: bool = False
        self._require_detections: bool = True
        self._can_plot: bool = True

    @abstractmethod
    def run(self):
        return None
    
    @property
    def require_detections(self):
        return self._require_detections
    
    @property
    def require_masks(self):
        return self._require_masks

    def set_ctx(self, imgs: List[str], detections: dict[str, ContextDetectionBoxData], masks: dict[str, ContextDetectionMaskData], data_classes: List[str] = None):
        self.set_images(imgs)
        self.set_detections(detections)
        self.set_masks(masks)
        self.set_data_classes(data_classes)

    def set_images(self, imgs: List[str]):
        self._ctx_imgs = imgs
    
    def set_detections(self, detections: dict[str, ContextDetectionBoxData]):
        self._ctx_detections = detections

    def set_masks(self, masks: dict[str, ContextDetectionMaskData]):
        self._ctx_masks = masks

    def set_data_classes(self, data_classes: List[str]):
        self._ctx_data_cls = data_classes

    @staticmethod
    def result_dataframe(img_cls_result: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
        """
        Get the numerical results for each class per image in a dictionary and transform it into a pandas DataFrame.
        'img_cls_result' format must be like: img_cls_result[img][class_label] = numerical_result_for_class_in_image
        """
        df_dict = {"img_idx": np.arange(0, len(img_cls_result))}
        cls_results = {}
        for i, img in enumerate(img_cls_result):
            for cls_label in img_cls_result[img]:
                if cls_label not in cls_results:
                    cls_results[cls_label] = np.zeros(len(df_dict["img_idx"]))
                cls_results[cls_label][i] += img_cls_result[img][cls_label]
        df_dict.update(cls_results)

        df = pd.DataFrame(df_dict)
        return df
