from typing import List, Union
import logging
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from object_detection_analysis.tasks import BaseAnalysisTask

class ObjectAreaOccupationTask(BaseAnalysisTask):
    def __init__(self, image_order: Union[List[int], None] = None,  use_boxes=False, plot=False, **plot_ax_kwargs):
        super().__init__()
        self._require_detections = True
        self._require_masks = not use_boxes
        self._can_plot = True

        self._config = {
            "image_order": image_order,
            "use_boxes": use_boxes,
            "plot": plot,
            "plot_ax_kwargs": plot_ax_kwargs,
        }
    
    def run(self):
        if (self._ctx_imgs is None) and (self._ctx_detections is None):
            logging.fatal("Trying to run Object Area Occupation Tasks but context detections and/or images weren't set")
            return None
        if not (self._config["use_boxes"]) and (self._ctx_masks is None):
            logging.fatal("Trying to run Object Area Occupation task using masks but no context masks were set")

        results = {
            img: { cls: 0 for cls in self._ctx_detections[img].object_classes } for img in self._ctx_detections
        }
        if not self._config["use_boxes"]: # use masks to calculate each leaf area
            for img, img_masks in self._ctx_masks.items():
                # Get total image pixels
                img_pixels = cv2.imread(img).size // 3
                
                # Store pixels occupied by objects
                results[img]["total"] = self.masks_pixels(img_masks.all_masks)
                for cls in img_masks.object_classes:
                    occupied_pixels = self.masks_pixels(img_masks.class_masks[cls])
                    results[img][cls] += occupied_pixels
                # Calculate each class density (including "total")
                for res in results[img]:
                    results[img][res] = results[img][res] / img_pixels
        else: # use detection box to approximate area
            for img, img_det in self._ctx_detections.items():
                # Total image pixels
                img_pixels = cv2.imread(img).size // 3
                # Calculate pixels occupied by object boxes
                results[img]["total"] = self.boxes_pixels(img_det.all_boxes)
                for cls in img_det.object_classes:
                    results[img][cls] += self.boxes_pixels(img_det.class_boxes[cls])
                # Calculate each class density (including "total")
                for res in results[img]:
                    results[img][res] = results[img][res] / img_pixels
       
        # Build data frame where first column is image_idx and other columns are the occupied area of each class in that image (including 'all')
        if self._config["image_order"] is None:
            image_idx = np.arange(0, len(self._ctx_imgs))
        else:
            image_idx = np.array(self._config["image_order"])
        df = self.result_dataframe(results)
        df["img_idx"] = image_idx

        return df
        
    @staticmethod
    def masks_pixels(binary_masks: np.ndarray):
        total_pixels = np.sum([mask.sum() for mask in binary_masks])
        return total_pixels
    
    @staticmethod
    def boxes_pixels(boxes: np.ndarray):
        total_pixels = np.sum([ ((x2-x1)*(y2-y1)) for [x1,y1,x2,y2] in boxes ])
        return total_pixels

