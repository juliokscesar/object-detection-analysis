from typing import Union, List
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import torch

import scg_detection_tools.utils.image_tools as imtools
import scg_detection_tools.utils.cvt as cvt

from object_detection_analysis.ctx_data import ContextDetectionBoxData, ContextDetectionMaskData
from object_detection_analysis.tasks import BaseAnalysisTask
from object_detection_analysis.classifiers import BaseClassifier, classifier_from_name

class ObjectClassificationTask(BaseAnalysisTask):
    """
    Classification of detected objects using one of the classifiers (object_detection_analysis.classifiers)
    The classificator must be a classification model pre-trained with the desired classes.
    """
    def __init__(
            self, 
            clf: Union[BaseClassifier, str], 
            clf_cls_colors: dict,
            clf_cls_labels: List[str],
            clf_ckpt_path: str = None, 
            clf_input_batch: int = 4,
            use_boxes: bool = False,
            show_classifications=True,
            show_detections=False,
            OBJ_STD_SIZE=(32,32),
            plot=False,
            **plot_ax_kwargs,
        ):
        """Initialize object classification task"""
        super().__init__()
        self._require_detections = True
        self._require_masks = not use_boxes
        self._can_plot = True
        
        if isinstance(clf, str):
            clf = classifier_from_name(clf, ckpt_path=clf_ckpt_path, to_optimal_device=True)
        self._clf = clf
        if self._clf is None:
            logging.fatal("ObjectClassificationTask classifier is missing")
            raise RuntimeError()
        
        self._config = {
            "clf_cls_colors": clf_cls_colors,
            "clf_cls_labels": clf_cls_labels,
            "clf_input_batch": clf_input_batch,
            "show_classifications": show_classifications,
            "use_boxes": use_boxes,
            "show_detections": show_detections, 
            "obj_std_size": OBJ_STD_SIZE,
            "plot": plot,
            "plot_ax_kwargs": plot_ax_kwargs,
        }

    def run(self):
        if (self._ctx_imgs is None) or (self._ctx_detections is None):
            logging.fatal("ObjectClassificationTask requires context images and detections but one of them is None.")
            return None
        if not (self._config["use_boxes"]) and (self._ctx_masks is None):
            logging.fatal("Object classification with 'use_boxes' disabled requires context masks to be set.")
            return None

        image_objects = self.extract_objects(self._ctx_detections, self._ctx_masks, use_boxes=self._config["use_boxes"], OBJ_STD_SIZE=self._config["obj_std_size"])
        clf_results = {}
        for img in image_objects:
            clf_results[img] = {}
            for label in self._config["clf_cls_labels"]:
                clf_results[img]["total"] = 0 
                clf_results[img][label] = 0

        # Prepare color patches for legend when showing results
        if self._config["show_classifications"]:
            color_patches = [
                mpatches.Patch(color=self._config["clf_cls_colors"][c], label=c) for c in self._config["clf_cls_labels"]
            ]

        PREDICTION_BATCH = self._config["clf_input_batch"]
        for img, obj_data in image_objects.items():
            # Run predictions in batches
            pred_cls = []
            for i in range(0, len(obj_data), PREDICTION_BATCH):
                obj_data_batch = [obj[2] for obj in obj_data[i:(i+PREDICTION_BATCH)]]
                pred_cls.extend(self._clf.predict(obj_data_batch))

            obj_label = [self._config["clf_cls_labels"][pred] for pred in pred_cls]
            clf_results[img]["total"] = len(obj_label)
            for cls in set(obj_label):
                clf_results[img][cls] = obj_label.count(cls)

            # Showing classifications
            if not self._config["show_classifications"]:
                continue
            _, axs = plt.subplots(ncols=2, figsize=(15,10), layout="tight")
            axs[0].axis("off")
            axs[1].axis("off")
            orig_img = cv2.imread(img)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            if self._config["show_detections"]:
                axs[0].imshow(cv2.cvtColor(imtools.box_annotated_image(img, boxes=self._ctx_detections[img].all_boxes, box_thickness=2), cv2.COLOR_BGR2RGB))
            else:
                axs[0].imshow(orig_img)
            
            ann_img = orig_img.copy()
            for idx in range(len(obj_label)):
                label = obj_label[idx]
                box, mask, _ = obj_data[idx]
                color = []
                if isinstance(self._config["clf_cls_colors"][label], str):
                    color = mcolors.to_rgb(self._config["clf_cls_colors"][label])
                else:
                    color = self._config["clf_cls_colors"][label]
                if mask is not None:
                    ann_img = imtools.apply_image_mask(ann_img, mask, color, alpha=0.6, bounding_box=box)
                else:
                    box_mask = cvt.boxes_to_masks([box], imghw=orig_img.shape[:2], binary_mask=True)[0]
                    box_mask = imtools.crop_box_image(box_mask, box)
                    ann_img = imtools.apply_image_mask(ann_img, box_mask, color, alpha=0.6, bounding_box=box)
            axs[1].imshow(ann_img)
            axs[1].legend(handles=color_patches)
            plt.show()

        df_results = self.result_dataframe(clf_results)
        return df_results

    @staticmethod
    def extract_objects(ctx_detections: dict[str, ContextDetectionBoxData], ctx_masks: dict[str, ContextDetectionMaskData] = None, use_boxes = False, OBJ_STD_SIZE=(32,32)):
        if (not use_boxes) and (ctx_masks is None):
            logging.fatal("Trying to extract objects with use_boxes disabled, but context masks is None")
            return None

        image_objects = {}
        for img in ctx_detections:
            image_objects[img] = []
            boxes = ctx_detections[img].all_boxes
            orig_img = cv2.imread(img)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            if not use_boxes:
                masks = ctx_masks[img].all_masks
                for box, mask in zip(boxes, masks):
                    obj_crop = imtools.crop_box_image(orig_img, box)
                    if obj_crop.shape[:2] != mask.shape[:2]:
                        mask = cv2.resize(mask, obj_crop.shape[:2], interpolation=cv2.INTER_CUBIC)
                    masked_obj = obj_crop * mask
                    masked_obj = cv2.resize(masked_obj, OBJ_STD_SIZE, interpolation=cv2.INTER_CUBIC)
                    image_objects[img].append([box, mask, masked_obj])
            else:
                for box in boxes:
                    obj_crop = imtools.crop_box_image(orig_img, box)
                    obj_crop = cv2.resize(obj_crop, OBJ_STD_SIZE, interpolation=cv2.INTER_CUBIC)
                    image_objects[img].append([box, None, obj_crop])
        return image_objects
