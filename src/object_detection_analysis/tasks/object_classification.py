from typing import Union, List
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

import scg_detection_tools.utils.image_tools as imtools

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
            use_boxes: bool = False,
            show_classifications=True,
            show_detections=False,
            plot_per_image=False,
            OBJ_STD_SIZE=(32,32),
        ):
        """Initialize object classification task"""
        super().__init__()
        self._require_detections = True
        self._require_masks = True
        
        if isinstance(clf, str):
            clf = classifier_from_name(clf, ckpt_path=clf_ckpt_path)
        self._clf = clf
        if self._clf is None:
            logging.fatal("ObjectClassificationTask classifier is missing")
            raise RuntimeError()
        
        self._config = {
            "clf_cls_colors": clf_cls_colors,
            "clf_cls_labels": clf_cls_labels,
            "show_classifications": show_classifications,
            "use_boxes": use_boxes,
            "show_detections": show_detections, 
            "plot_per_image": plot_per_image,
            "obj_std_size": OBJ_STD_SIZE,
        }

    def run(self):
        if (self._ctx_imgs is None) or (self._ctx_detections is None):
            logging.fatal("ObjectClassificationTask requires context images and detections but one of them is None.")
            return None
        if not (self._config["use_boxes"]) and (self._ctx_masks is None):
            logging.fatal("Object classification with 'use_boxes' disabled requires context masks to be set.")

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

        for img, obj_data in image_objects.items():
            pred_cls = self._clf.predict([obj[2] for obj in obj_data])
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
            
            ann_img = orig_img
            for idx in range(len(obj_label)):
                label = obj_label[idx]
                box, mask, _ = obj_data[idx]
                color = []
                if isinstance(self._config["clf_cls_colors"][label], str):
                    color = mcolors.to_rgb(self._config["clf_cls_colors"][label])
                else:
                    color = self._config["clf_cls_colors"][label]
                if mask is not None:
                    ann_img = imtools.segment_annotated_image(ann_img, mask, color, alpha=0.6)
                else:
                    ann_img = imtools.box_annotated_image(ann_img, [box])
            axs[1].imshow(ann_img)
            axs[1].legend(handles=color_patches)
            plt.show()

        # Plot each class count per image
        if self._config["plot_per_image"]:
            _, ax = plt.subplots(layout="tight")
            img_idx = np.arange(len(self._ctx_imgs))
            cls_count = {}
            for img in clf_results:
                for cls in clf_results[img]:
                    if cls == "total":
                        continue
                    if cls not in cls_count:
                        cls_count[cls] = []
                    cls_count[cls].append(clf_results[img][cls])
            for cls in cls_count:
                ax.plot(img_idx, cls_count[cls], marker='o', label=cls)
            ax.legend()
            ax.set(xlabel="Image ID", ylabel="Object count")
            plt.show()

        return clf_results

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
                    h, w = mask.shape[1:]
                    mask = mask.astype(np.uint8).reshape(h,w,) * 255
                    masked = orig_img.copy()
                    masked[mask[:,:] < 255] = 0
                    obj_crop = imtools.crop_box_image(masked, box)
                    obj_crop = cv2.resize(obj_crop, OBJ_STD_SIZE, cv2.INTER_CUBIC)
                    image_objects[img].append([box, mask, obj_crop])
            else:
                for box in boxes:
                    obj_crop = imtools.crop_box_image(orig_img, box)
                    obj_crop = cv2.resize(obj_crop, OBJ_STD_SIZE, cv2.INTER_CUBIC)
                    image_objects[img].append(
                        [box, None, obj_crop]
                    )
        return image_objects
