import logging
from typing import Union
from enum import Enum, auto
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from object_detection_analysis.tasks import BaseAnalysisTask
from object_detection_analysis.ctx_data import ContextDetectionBoxData, ContextDetectionMaskData

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
        hist_area_bins = 30,
        hist_shape_bins = 30,
        hist_diagonal_bins = 30,
    ):
        super().__init__()
        self._require_detections = True
        self._require_masks = True
        self._can_plot = False

        if isinstance(fit_shape, str):
            fit_shape = FittableShape[fit_shape.strip().upper()]

        self._config = {
            "calc_area": area,
            "hist_area_bins": hist_area_bins,

            "calc_box_diagonal": box_diagonal,
            "hist_diagonal_bins": hist_diagonal_bins,

            "fit_shape": fit_shape,
            "hist_shape_bins": hist_shape_bins,

            "show_histograms": show_histograms,
        }

    def run(self):
        if (self._ctx_detections is None) or (self._ctx_masks is None):
            logging.fatal("Trying to run Shape Properties task but context detections and/or masks is None")
            return None
        
        results = {}
        for img in self._ctx_detections:
            basename = os.path.basename(img)
            if img not in results:
                results[img] = {}
            
            if self._config["calc_box_diagonal"]:
                results[img]["box_diagonal"] = self.objects_diagonal(self._ctx_detections[img])
                if self._config["show_histograms"]:
                    print("_"*90)
                    print(f"\t BOX DIAGONAL RESULTS FOR IMAGE {basename!r}")
                    self.objects_show_histogram(results[img]["box_diagonal"], bins=self._config["hist_diagonal_bins"], data_type="box_diagonal", title=basename)
            
            if self._config["calc_area"]:
                if img not in self._ctx_masks:
                    logging.error(f"Calculating area of each object requires masks to be set. But masks for image {img} is missing")
                    continue
                results[img]["area"] = self.objects_area(self._ctx_masks[img])
                if self._config["show_histograms"]:
                    print("_"*90)
                    print(f"\t AREA RESULTS FOR IMAGE {basename!r}")
                    self.objects_show_histogram(results[img]["area"], bins=self._config["hist_area_bins"], data_type="area", title=basename)

            if self._config["fit_shape"]:
                if img not in self._ctx_masks:
                    logging.error(f"Calculating area of each object requires masks to be set. But masks for image {img} is missing")
                    continue
                results[img]["fit_shape"] = self.objects_fitshape(self._ctx_masks[img], self._config["fit_shape"])
                if self._config["show_histograms"]:
                    print("_"*90)
                    print(f"\t FIT SHAPE RESULTS FOR IMAGE {basename!r}")
                    data_type = "ellipsis_shape" if self._config["fit_shape"] == FittableShape.ELLIPSIS else "circle_shape"
                    self.objects_show_histogram(results[img]["fit_shape"], bins=self._config["hist_shape_bins"], data_type=data_type, title=basename)
        
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

    @staticmethod
    def objects_show_histogram(data: np.ndarray, bins: int, data_type: str = "area", title: Union[str,None] = None):
        if data_type == "area":
            _, ax = plt.subplots(layout="tight")
            ax.set(xlabel="Individual Area", ylabel="Frequency")
            ax.hist(data, bins=bins)
        
        elif data_type == "box_diagonal":
            _, ax = plt.subplots(layout="tight")
            ax.set(xlabel="Box Diagonal", ylabel="Frequency")
            ax.hist(data, bins=bins)
        
        elif data_type == "circle_shape":
            _, ax = plt.subplots(layout="tight")
            ax.set(xlabel="Fitted circle radius", ylabel="Frequency")
            ax.hist(data.ravel(), bins=bins)
    
        elif data_type == "ellipsis_shape":
            # Plot 3D histogram for ellipsis shape
            a_values = [shape[0] for shape in data]
            b_values = [shape[1] for shape in data]

            hist, xedges, yedges = np.histogram2d(a_values, b_values, bins=bins)
            xpos, ypos = np.meshgrid(xedges[:-1]+0.5, yedges[:-1]+0.5, indexing="ij")
            xpos = xpos.ravel()
            ypos = ypos.ravel()
            zpos = np.zeros_like(xpos)

            dx = dy = 0.5
            dz = hist.ravel()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort="average")
            ax.set(xlabel="Ellipsis Major Axis (a)", ylabel="Ellipsis Minor Axis (b)", zlabel="Frequency")

        if title is not None:
            ax.set_title(title)
        
        plt.show()