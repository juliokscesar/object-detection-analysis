from abc import ABC, abstractmethod
from typing import List, Union
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cv2

from scg_detection_tools.utils.file_handling import get_annotation_files
from scg_detection_tools.dataset import read_dataset_annotation
import scg_detection_tools.utils.image_tools as imtools
from classifers import BaseClassifier, classifier_from_name

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


class CountValidateModelTask(BaseAnalysisTask):
    def __init__(self, annotations_path: str, export_csv = False):
        super().__init__()
        self._config = {
            "export_csv": export_csv,
            "annotations_path": annotations_path,
        }
        self._require_detections = True
        self._require_masks = False

    def run(self):
        if not self._load_annotations():
            return None
        if (self._ctx_detections is None) or (self._ctx_data_cls is None):
            logging.error("CountValidateModelTask requires context images detections and data classes")
            return None
        
        # Not interested in errors per each class, only in the total count
        true_count = []
        for img in self._ctx_imgs:
            if self._annotation_files[img] is None:
                true_count.append(0)
            else:
                ann = read_dataset_annotation(self._annotation_files[img], separate_class=False)
                true_count.append(len(ann) - 1)
        true_count = np.array(true_count)
        pred_count = [
            CountAnalysisTask._count_per_cls(self._ctx_detections[img])["all"] for img in self._ctx_detections
        ]
        pred_count = np.array(pred_count)
        result_errors = self.calculate_errors(true_count, pred_count)
        print(f"OBJECT COUNT VALIDATION RESULTS:")
        for metric in result_errors:
            print(f"\t{metric.upper()}: {result_errors[metric]}")
        return result_errors

    @staticmethod
    def calculate_errors(true_count: np.ndarray, pred_count: np.ndarray):
        if not isinstance(true_count, np.ndarray):
            true_count = np.array(true_count)
        if not isinstance(pred_count, np.ndarray):
            pred_count = np.array(pred_count)
        if true_count.shape != pred_count.shape:
            logging.error(f"'true_count' (shape={true_count.shape}) and 'pred_count' (shape={pred_count.shape}) have different shapes. They must be the same in order to calculate the errors")
            return None
        
        errors = true_count - pred_count
        relative = np.divide(
            errors.astype(np.float64),
            true_count.astype(np.float64),
            out=np.zeros_like(errors.astype(np.float64)),
            where=true_count != 0,
        )

        mae = np.abs(errors).mean()
        mse = (errors**2).mean()
        rmse = np.sqrt(mse)
        stderror = np.sqrt( ((errors - errors.mean())**2).mean() )

        rel_mae = np.absolute(relative).mean()
        rel_mse = (relative**2).mean()
        rel_rmse = np.sqrt(rel_mse)
        rel_stderror = np.sqrt( ((relative - relative.mean())**2).mean() )

        return {
            "mae": mae, "mse": mse, "rmse": rmse, "stderror": stderror,
            "relative_mae": rel_mae, "relative_mse": rel_mse, "relative_rmse": rel_rmse, "relative_stderror": rel_stderror,
        }
            
    def _load_annotations(self):
        if self._ctx_imgs is None:
            logging.error("Trying to load annotations for validating model but no images from context were set.")
            return False
        self._annotation_files = get_annotation_files(self._ctx_imgs, self._config["annotations_path"])
        return True


class CountAnalysisTask(BaseAnalysisTask):
    def __init__(self, export_csv=False, plot_all=True, plot_per_class=True, save_plots=False):
        super().__init__()
        self._config = {
            "export_csv": export_csv,
            "plot_all": plot_all,
            "plot_per_class": plot_per_class,
            "save_plots": save_plots,
        }
        self._require_detections = True
        self._require_masks = False
    
    def run(self):
        # Check if any data missing
        if (self._ctx_imgs is None) or (self._ctx_detections is None):
            logging.error("Trying to run CountAnalysisTask but either context images or detections weren't set.")
            return None

        result = {
            img: self._count_per_cls(self._ctx_detections[img]) for img in self._ctx_detections
        }
        # Make a dictionary for dataframe with columns "img_idx" and class names
        img_idx = np.arange(len(self._ctx_imgs))
        data = {"img_idx": img_idx}
        for i, img in enumerate(result):
            for cls in result[img]:
                if cls not in data:
                    data[cls] = np.zeros(len(img_idx))
                data[cls][i] += result[img][cls]        
        df = pd.DataFrame(data)
        df = df.set_index("img_idx")

        if self._config["export_csv"]:
            df.to_csv("count_analysis_task_out.csv")

        fig_plots = []
        if self._config["plot_all"]:
            fig_all, ax = plt.subplots(layout="tight")
            ax.plot(data["img_idx"], data["all"], marker='o')
            ax.set(xlabel="Image ID", ylabel="Object count")
            fig_plots.append(("plot_all", fig_all))

        if self._config["plot_per_class"]:
            fig_pcls, ax = plt.subplots(layout="tight")
            for cls in list(data.keys())[1:]:
                ax.plot(data["img_idx"], data[cls], marker='o', label=cls)
            ax.legend()
            ax.set(xlabel="Image ID", ylabel="Object count")
            fig_plots.append(("plot_per_class", fig_pcls))

        if self._config["save_plots"]:
            for name, fig in fig_plots:
                fig.savefig(f"analysis_exp/plots/count_{name}.png")

        plt.show()
        return result

    @staticmethod
    def _count_per_cls(cls_detections):
        results = {"all": 0}
        for cls in cls_detections:
            if cls not in results:
                results[cls] = len(cls_detections[cls])
            else:
                results[cls] += len(cls_detections[cls])
            results["all"] += len(cls_detections[cls])
        return results
            

class ObjectClassificationTask(BaseAnalysisTask):
    def __init__(
            self, 
            clf: Union[BaseClassifier, str], 
            clf_ckpt_path: str = None, 
            clf_cls_colors: dict = None,
            clf_cls_labels: List[str] = None,
            show_classifications=True,
            show_detections=False,
            plot_per_image=False,
            OBJ_STD_SIZE=(32,32),
        ):
        super().__init__()
        
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
            "show_detections": show_detections, 
            "plot_per_image": plot_per_image,
            "obj_std_size": OBJ_STD_SIZE,
        }

    def run(self):
        if (self._ctx_imgs is None) or (self._ctx_detections is None) or (self._ctx_masks is None):
            logging.fatal("ObjectClassificationTask requires context images, detections and masks but one of them is None.")
            return None

        image_objects = self.extract_objects(self._ctx_detections, self._ctx_masks, OBJ_STD_SIZE=self._config["obj_std_size"])
        clf_results = {img: {"total": 0} for img in image_objects}
        for img, obj_data in image_objects.items():
            pred_cls = self._clf.predict([obj[2] for obj in obj_data])
            if self._config["clf_cls_labels"] is not None:
                obj_label = [self._config["clf_cls_labels"][pred] for pred in pred_cls]
            else:
                obj_label = [str(c) for c in pred_cls]
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
                axs[0].imshow(cv2.cvtColor(imtools.box_annotated_image(img, self._ctx_detections[img]["all"], box_thickness=2), cv2.COLOR_BGR2RGB))
            else:
                axs[0].imshow(orig_img)
            
            ann_img = orig_img
            for idx in range(len(obj_label)):
                label = obj_label[idx]
                _, mask, _ = obj_data[idx]
                color = []
                if self._config["clf_cls_colors"] is None:
                    color = np.random.randint(low=0, high=255, size=3)
                else:
                    if isinstance(self._config["clf_cls_colors"][label], str):
                        color = mcolors.to_rgb(self._config["clf_cls_colors"][label])
                    else:
                        color = self._config["clf_cls_colors"][label]
                ann_img = imtools.segment_annotated_image(ann_img, mask, color, alpha=0.6)
            axs[1].imshow(orig_img)
            # TODO: axs[1].legend()
            plt.show()

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
    def extract_objects(ctx_detections: dict, ctx_masks: dict, OBJ_STD_SIZE=(32,32)):
        image_objects = {}
        for (img, cls_detections) in ctx_detections.items():
            image_objects[img] = []
            boxes = cls_detections["all"]
            masks = ctx_masks[img]["all"]
            orig_img = cv2.imread(img)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

            for box, mask in zip(boxes, masks):
                h, w = mask.shape[1:]
                mask = mask.astype(np.uint8).reshape(h,w,)
                #mask = np.where(mask == 1, 255, 0)
                mask = mask * 255
                orig_img[mask[:,:] < 255] = 0
                obj_crop = imtools.crop_box_image(orig_img, box)
                obj_crop = cv2.resize(obj_crop, OBJ_STD_SIZE, cv2.INTER_CUBIC)
                image_objects[img].append([box, mask, obj_crop])
        return image_objects                
