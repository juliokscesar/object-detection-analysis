from typing import List, Union
import logging
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from object_detection_analysis.tasks import BaseAnalysisTask

class ObjectAreaOccupationTask(BaseAnalysisTask):
    def __init__(self, image_order: Union[List[int], None] = None,  export_csv=False, plot=True, plot_per_class=False):
        super().__init__()
        self._require_detections = True
        self._require_masks = True

        self._config = {
            "image_order": image_order,
            "plot": plot,
            "plot_per_class": plot_per_class,
            "export_csv": export_csv,
        }
    
    def run(self):
        if (self._ctx_masks is None) or (self._ctx_imgs is None):
            logging.fatal("Trying to run Object Area Occupation Tasks but context masks and/or images weren't set")
            return None

        results = {
            img: { cls: 0 for cls in self._ctx_detections[img].object_classes } for img in self._ctx_detections
        }
        for img, img_masks in self._ctx_masks.items():
            # Get total image pixels
            img_pixels = cv2.imread(img).size // 3
            
            # Store pixels occupied by objects
            results[img]["all"] = self.masks_pixels(img_masks.all_masks)
            for cls in img_masks.object_classes:
                occupied_pixels = self.masks_pixels(img_masks.class_masks[cls])
                results[img][cls] += occupied_pixels
            # Calculate each class density (including "all")
            for res in results[img]:
                results[img][res] = results[img][res] / img_pixels
        
        if not self._config["plot"] and not self._config["plot_per_class"] and not self._config["export_csv"]:
            return results
        
        # Build data frame where first column is image_idx and other columns are the occupied area of each class in that image (including 'all')
        if self._config["image_order"] is None:
            image_idx = np.arange(0, len(self._ctx_imgs))
        else:
            image_idx = np.array(self._config["image_order"])
        data = {"image_idx": np.arange(1, len(self._ctx_masks) + 1)}
        for idx in image_idx:
            img = self._ctx_imgs[idx]
            for cls in results[img]:
                if cls not in data:
                    data[cls] = np.zeros(len(image_idx))
                data[cls][idx] = results[img][cls]
        df = pd.DataFrame(data)
        df = df.set_index("image_idx")

        if self._config["export_csv"]:
            df.to_csv("area_occupation.csv")
        
        if self._config["plot"]:
            _, ax = plt.subplots(layout="tight")
            ax.plot(df.index.values, df["all"], marker='o')
            ax.set(xlabel="Image index", ylabel="Area occupied by all objects")
            plt.show()
        
        if self._config["plot_per_class"]:
            _, ax = plt.subplots(layout="tight")
            for cls in df.columns:
                if cls == "all":
                    continue
                ax.plot(df.index.values, df[cls], marker='o', label=cls)
            ax.legend()
            ax.set(xlabel="Image index", ylabel="Area occupied by objects of the same class")
            plt.show()

        return results
        
    @staticmethod
    def masks_pixels(binary_masks: np.ndarray):
        total_pixels = np.sum([mask.sum() for mask in binary_masks])
        return total_pixels

