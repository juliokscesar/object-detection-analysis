import logging
from typing import Union
from enum import Enum, auto
import numpy as np
import cv2

from object_detection_analysis import BaseAnalysisTask, ContextDetectionMaskData, ContextDetectionBoxData

class FittableShape(Enum):
    CIRCLE = auto()
    ELLIPSIS = auto()

class ShapePropertiesTask(BaseAnalysisTask):
    """
    Analyze properties of the object based on its shape by using its segmented mask
    """
    def __init__(
        self,
        fit_shape: Union[str, FittableShape, None] = None,
        area = True,
        box_diagonal = True,
        show_histograms = True,
    ):
        super().__init__()
        self._require_detections = True
        self._require_masks = True
        self._can_plot = False

        if isinstance(fit_shape, str):
            fit_shape = FittableShape[fit_shape.strip().upper()]

        self._config = {
            "calc_area": area,
            "calc_box_diagonal": box_diagonal,
            "fit_shape": fit_shape,
            "show_histograms": show_histograms,
        }

    def run(self):
        if (self._ctx_detections is None) or (self._ctx_masks is None):
            logging.fatal("Trying to run Shape Properties task but context detections and/or masks is None")
            return None
        
        results = {}
        for img in self._ctx_detections:
            if img not in results:
                results[img] = {}
            
            if self._config["calc_box_diagonal"]:
                results[img]["box_diagonal"] = self.objects_diagonal(self._ctx_detections[img])
            
            if self._config["calc_area"]:
                if img not in self._ctx_masks:
                    logging.error(f"Calculating area of each object requires masks to be set. But masks for image {img} is missing")
                results[img]["area"] = self.objects_area(self._ctx_masks[img])
        
        return results
            
    @staticmethod
    def objects_area(ctx_masks: ContextDetectionMaskData) -> np.ndarray:
        areas = np.array([
            mask.sum() for mask in ctx_masks.all_masks
        ])
        return areas

    @staticmethod
    def objects_diagonal(ctx_boxes: ContextDetectionBoxData) -> np.ndarray:
        diagonals = np.array([
            np.sqrt((box[2]-box[0])**2.0 + (box[3]-box[1])**2.0) for box in ctx_boxes.all_boxes
        ])
        return diagonals

    @staticmethod
    def objects_fitshape(ctx_masks: ContextDetectionMaskData, shape: FittableShape) -> np.ndarray:
        shape_props = []
        if shape == FittableShape.ELLIPSIS:
            for mask in ctx_masks.all_masks:
                contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                contour = max(contours, key=cv2.contourArea)
                ellipse = cv2.fitEllipse(contour)
                a = ellipse[1][0]/2 # get 'a' parameter from major-axis
                b = ellipse[1][1]/2 # get 'b' parameter from minor-axis
                shape_props.append((a,b))
        elif shape == FittableShape.CIRCLE:
            for mask in ctx_masks.all_masks:
                contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                contour = max(contours, key=cv2.contourArea)
                _, radius = cv2.minEnclosingCircle(contour)
                shape_props.append((radius))
        return np.array(shape_props)