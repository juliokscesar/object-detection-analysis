from typing import Union, List, Tuple
import logging
import numpy as np
import pandas as pd
import torch
import cv2
import os
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
import itertools

from scg_detection_tools.models import BaseDetectionModel, get_opt_device, from_type
from scg_detection_tools.detect import Detector
from scg_detection_tools.utils.file_handling import read_yaml, file_exists, clear_temp_folder
import scg_detection_tools.utils.image_tools as imtools

from object_detection_analysis.ctx_data import ContextDetectionBoxData, ContextDetectionMaskData
from object_detection_analysis.tasks import BaseAnalysisTask

DEFAULT_ANALYSIS_CONFIG = {
    "detection_parameters": {
        "confidence": 50.0,
        "overlap": 50.0,
        "slice_detect": True,
        "slice_wh": (640,640),
        "slice_overlap_ratio": (0.3, 0.3),
        "slice_iou_threshold": 0.2,
        "slice_fill": False,

        "detection_filters": {
            "duplicate_filter": True,
            "duplicate_filter_thresh": 0.95,
            "object_size_filter": True,
            "object_size_max_wh": (80,80),
        },
        
        "enable_image_preprocess": False,
        "image_preprocess": { 
            "apply_to_ratio": 1.0,
            "parameters": {"contrast_ratio": 1.0,"brightness_delta": 0} ,
        },

    },

    "use_specific_parameters": False,
    "image_specific_parameters": None,

    "model_type": "yolov8",
    "model_path": "yolov10l.pt",
    "data_classes": ["leaf"],
    "sam2_path": "sam2_hiera_tiny.pt",
    "sam2_cfg": "sam2_hiera_t.yaml",
}


class DetectionAnalysisContext:
    """
    Base class to use for analysis. This holds information about which tasks to run and all the buffer data
    including which images to use, their detections, masks and anything else that may be used.
    """
    def __init__(
            self, imgs: List[str], 
            det_model: BaseDetectionModel = None, 
            config: Union[str, dict] = DEFAULT_ANALYSIS_CONFIG,
            image_specific_parameters: Union[dict[int,dict], dict[str,dict]] = None,
            tasks: Tuple[str, List[BaseAnalysisTask]] = None,
        ):
        # Read from yaml if string passed
        if isinstance(config, str):
            config = read_yaml(config)
        self._config = {
            key: config[key] if key in config else DEFAULT_ANALYSIS_CONFIG[key] for key in DEFAULT_ANALYSIS_CONFIG 
        }
        if image_specific_parameters is not None:
            self._config["image_specific_parameters"] = image_specific_parameters
            self._config["use_specific_parameters"] = True

        if det_model is None:
            det_model = from_type(config["model_type"], config["model_path"], config["data_classes"])
        self._detector = Detector(det_model, detection_params=config["detection_parameters"].copy())
        # self._segmentor = SAM2Segment(config["sam2_path"], config["sam2_cfg"])
        self._segmentor = None

        self._ctx_device = get_opt_device()

        # Initialize context data
        self._ctx_imgs = imgs
        self._ctx_detections = {}
        self._ctx_masks = {}
        for img in imgs:
            self._ctx_detections[img] = ContextDetectionBoxData(
                object_classes=self._config["data_classes"],
                all_boxes=[],
                class_boxes={cls: [] for cls in self._config["data_classes"]},
            )
            self._ctx_masks[img] = ContextDetectionMaskData(
                object_classes=self._config["data_classes"],
                all_masks=[],
                class_masks={cls: [] for cls in self._config["data_classes"]},
            )
        self._detect = self._segment = False 
        self._tasks = None
        self._tasks_results = None
        if tasks is not None:
            self._tasks = {}
            self._tasks_results = {}
            for name, task in tasks:
                self._tasks[name] = task
                self._tasks_results[name] = None

                if (not self._detect) and (task.require_detections):
                    self._detect = True
                if (not self._segment) and (task.require_masks):
                    self._segment = True


    @property 
    def images(self):
        return self._ctx_imgs

    def add_image(self, img: str):
        if not file_exists(img):
            raise ValueError(f"File {img!r} does not exist.")
        self._ctx_imgs.append(img)

    def change_configs(self, config: dict):
        for key in config:
            if key not in self._config:
                raise ValueError(f"Key {key} not in Detection Analysis Context config.")
            # Update detector if detection_parameters changed, and update config only changing the keys passed
            if key == "detection_parameters":
                self._detector.update_parameters(**config[key])
                self._config["detection_parameters"] = self._detector._det_params.copy()
            else:
                self._config[key] = config[key]

    @property
    def tasks(self):
        return self._tasks
    
    def add_task(self, name: str, task: BaseAnalysisTask):
        self._tasks.append((name, task))

    def create_subcontext(self):
        return DetectionAnalysisContext(
            imgs = self.images,
            det_model=self._detector._det_model,
            config=self._config,
        )

    def run(self, clear_temp=True):
        if self._detect:
            # run detection and free memory by deleting variables
            self._run_detections()
            self._detect = False
        if self._segment:
            # run segmentation and free mmemory by deleting variables
            self._run_segmentations()
            self._segment = False
        for name, task in self._tasks.items():
            logging.info(f"Starting task {name}")
            task.set_ctx(
                imgs=self._ctx_imgs,
                detections=self._ctx_detections,
                masks=self._ctx_masks,
                data_classes=self._config["data_classes"],
            )
            task_result = task.run()
            self._tasks_results[name] = task_result
            logging.info(f"Finished task {name}")
        
        if clear_temp:
            clear_temp_folder()

    def _run_detections(self):
        logging.info("Started running detections with context images")
        images = self.images.copy()
        # Run any detections that uses specific parameters
        if self._config["use_specific_parameters"]:
            track_detected_imgs = []
            for img in self._config["image_specific_parameters"]:
                self._detector.update_parameters(**self._config["image_specific_parameters"][img])
                print("USING SPECIFIC PARAMETERS FOR IMAGE", img, self._detector._det_params)
                if isinstance(img, int):
                    img = images[img]
                detections = self._detector(img)[0]
                for cls_id, box in zip(detections.class_id, detections.xyxy.astype(np.int32)):
                    cls_label = self._config["data_classes"][cls_id]
                    self._ctx_detections[img].class_boxes[cls_label].append(box)
                    self._ctx_detections[img].all_boxes.append(box)
                track_detected_imgs.append(img)
            for img in track_detected_imgs:
                images.remove(img)
            # reset detection parameters
            self._detector.update_parameters(**self._config["detection_parameters"])

        # Run rest of detections with common parameters
        detections = self._detector(images)
        for img, detection in zip(images, detections):
            for cls_id, box in zip(detection.class_id, detection.xyxy.astype(np.int32)):
                if cls_id > len(self._config["data_classes"]):
                    logging.warning(f"DetectionAnalysisContext._run_detections: class id '{cls_id}' in detections from image {img} not in context data classes. Skipping...")
                    continue
                cls_label = self._config["data_classes"][cls_id]
                self._ctx_detections[img].class_boxes[cls_label].append(box)
                self._ctx_detections[img].all_boxes.append(box)
        logging.info("Finished running detections")
        self._ctx_detections = dict(sorted(self._ctx_detections.items(), key=lambda t: int(Path(t[0]).stem) if Path(t[0]).stem.isnumeric() else t[0]))
        torch.cuda.empty_cache()

    def _run_segmentations(self):
        # It is necessary to keep this import here in order to make YOLO-NAS work with SAM2
        # because importing both together conflicts the initialization of Hydra module
        # which both depend on.
        from scg_detection_tools.segment import SAM2Segment

        logging.info("Started running segmentation using detections from context images")

        if self._segmentor is None:
            self._segmentor = SAM2Segment(sam2_ckpt_path=self._config["sam2_path"], sam2_cfg=self._config["sam2_cfg"])

        # Check for any image with no detections in context data
        no_detections = list(set(self._ctx_imgs).difference(self._ctx_detections.keys()))
        if len(no_detections) > 0:
            logging.warning(f"No detections in context for images {', '.join([f'{img!r}' for img in no_detections])}")

        for img, cls_detections in self._ctx_detections.items():
            if len(cls_detections.all_boxes) == 0:
                logging.warning(f"Trying to segment boxes from image {img!r} but it has 0 detection boxes in context. Skipping...")
                continue
            orig_img = cv2.imread(img)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            for cls in cls_detections.class_boxes:
                # Instead of keeping an entire mask image for the object, keep just a mask of the crop. Then use the box+mask to show
                # Also, pass in a batch image of the objects instead of passing one by one
                cls_masks = []
                OBJ_SEGMENTATION_BATCH = 64
                OBJ_STD_SIZE = (32,32)
                for obj_idx in range(0, len(cls_detections.class_boxes[cls]), OBJ_SEGMENTATION_BATCH):
                    # Get all the bounding boxes on the image for this batch
                    box_batch = np.array(cls_detections.class_boxes[cls][obj_idx:(obj_idx+OBJ_SEGMENTATION_BATCH)])
                    obj_crop = [
                        cv2.resize(imtools.crop_box_image(orig_img, box), OBJ_STD_SIZE, interpolation=cv2.INTER_CUBIC) for box in box_batch
                    ]
                    # batch will be (rows x cols)
                    rows = cols = int(np.sqrt(OBJ_SEGMENTATION_BATCH))
                    batchh,batchw = rows*OBJ_STD_SIZE[0], cols*OBJ_STD_SIZE[1]
                    batch_img = np.zeros(shape=(batchh, batchw, 3), dtype=np.uint8)
                    batch_boxes = []
                    batch_obj_grid = np.full(shape=(rows, cols), fill_value=-1, dtype=np.int32)
                    which_obj = 0
                    for row in range(rows):
                        for col in range(cols):
                            initrow, finalrow = (row*OBJ_STD_SIZE[0]), ((row+1)*OBJ_STD_SIZE[0])
                            initcol, finalcol = (col*OBJ_STD_SIZE[1]), ((col+1)*OBJ_STD_SIZE[1])
                            
                            if which_obj >= len(obj_crop):
                                break
                            obj = deepcopy(obj_crop[which_obj])

                            batch_img[initrow:finalrow, initcol:finalcol] = obj

                            batch_boxes.append([initcol, initrow, finalcol, finalrow])
                            # add the box index to the grid
                            batch_obj_grid[row, col] = which_obj
                            which_obj += 1
                    batch_boxes = np.array(batch_boxes)

                    batch_masks = self._segmentor.segment_boxes(batch_img, batch_boxes)
                    mask_to_obj_idx = []
                    obj_masks = []
                    for mask_idx in range(len(batch_masks)):
                        # Batch masks comes with every mask but iterating through every column instead of every row
                        # So the mask grid is the transposed box grid
                        mask_row = mask_idx // rows
                        mask_col = mask_idx % cols

                        h, w = batch_masks[mask_idx].shape[-2:]
                        obj_mask = batch_masks[mask_idx].astype(np.uint8).reshape(h,w,1)

                        batch_obj = batch_obj_grid[mask_row,mask_col]
                        batch_bounding_box = batch_boxes[batch_obj]
                        obj_mask = imtools.crop_box_image(obj_mask, batch_bounding_box)

                        obj_box = box_batch[batch_obj]
                        obj_size = (obj_box[2]-obj_box[0], obj_box[3]-obj_box[1])
                        obj_mask = cv2.resize(obj_mask, obj_size, interpolation=cv2.INTER_CUBIC)
                        obj_masks.append(obj_mask.reshape(obj_mask.shape[0], obj_mask.shape[1], 1))
                        mask_to_obj_idx.append(batch_obj)
                    cls_masks.extend([obj_masks[i] for i in mask_to_obj_idx])

                self._ctx_masks[img].class_masks[cls] = cls_masks
                self._ctx_masks[img].all_masks.extend(cls_masks)
        torch.cuda.empty_cache()

    def task_result(self, task_name):
        if task_name not in self._tasks_results:
            logging.error(f"No results for task {task_name!r}")
            return None
        return self._tasks_results[task_name]

    def task_summary(self):
        print("-"*30, "TASK SUMMARY OF LAST ANALYSIS", "-"*30, "\n")
        for task in self._tasks_results:
            print(f"RESULTS OF TASK {task}")
            print(self._tasks_results[task])
            print("_"*90, "\n")

    def plot_tasks(self, plot_total_results=True, plot_per_class_results=False, save_plots=False):
        if (not plot_total_results) and (not plot_per_class_results):
            logging.warning("'plot_tasks' called with both plot_total_results and plot_per_class_results set to False. Nothing to plot then.")
            return

        if save_plots and (not os.path.isdir("analysis_exp/plots")):
            os.makedirs("analysis_exp/plots", exist_ok=True)

        for task_name, task_results in self._tasks_results.items():
            if (not self._tasks[task_name]._can_plot) or (not self._tasks[task_name]._config["plot"]):
                continue
            # make sure we're using dataframes
            if not isinstance(task_results, pd.DataFrame):
                logging.error(f"Expected type pd.DataFrame, but task results for task {task_name!r} is {type(task_results)}")
                continue
                
            if plot_total_results:
                fig, ax = plt.subplots(layout="tight")
                ax.plot(task_results["img_idx"], task_results["total"], marker='o')
                ax.set(xlabel="Image ID", ylabel=f"{task_name} total results")
                if len(self._tasks[task_name]._config["plot_ax_kwargs"]) > 0:
                    ax.set(**self._tasks[task_name]._config["plot_ax_kwargs"])
                if save_plots:
                    fig.savefig(f"analysis_exp/plots/task_{task_name}_total.png")

            if plot_per_class_results:
                fig, ax = plt.subplots(layout="tight")
                marker_iter = itertools.cycle(['o', '*', '.', '^', 'p', '+', 'x', 'D'])
                for cls in task_results.columns[1:]:
                    if cls == "total":
                        continue
                    ax.plot(task_results["img_idx"], task_results[cls], marker=next(marker_iter), label=cls)
                ax.set(xlabel="Image ID", ylabel=f"{task_name} per class results")
                if len(self._tasks[task_name]._config["plot_ax_kwargs"]) > 0:
                    ax.set(**self._tasks[task_name]._config["plot_ax_kwargs"])
                ax.legend()
                if save_plots:
                    fig.savefig(f"analysis_exp/plots/task_{task_name}_per_class.png")

    def export_csv_tasks(self, skip_tasks: List[str] = None):
        if not os.path.isdir("analysis_exp"):
            os.makedirs("analysis_exp", exist_ok=True)
        for task_name, task_results in self._tasks_results.items():
            if (skip_tasks is not None) and (task_name in skip_tasks):
                continue
            if not isinstance(task_results, pd.DataFrame):
                logging.error(f"Expected type pd.DataFrame, but task results for task {task_name!r} is {type(task_results)}")
                continue
            task_results.to_csv(f"analysis_exp/{task_name}.csv", index=False)

    def show_detections(self, only_imgs: Union[List[int], List[str], None] = None):
        if only_imgs is None:
            only_imgs = self.images
        elif isinstance(only_imgs[0], int):
            only_imgs = [self.images[i] for i in only_imgs]
        elif not isinstance(only_imgs[0], str):        
            logging.error("Argument 'only_imgs' must be either None, a list of images indices (List[int]) or a list of images paths (List[str])")
            return
        
        for img in only_imgs:
            if img not in self._ctx_detections:
                imtools.plot_image(cv2.imread(img))
                continue
            imtools.plot_image_detection(img, boxes=self._ctx_detections[img].all_boxes)

    def show_masks(self, only_imgs: Union[List[int], List[str], None] = None):
        if only_imgs is None:
            only_imgs = self.images
        elif isinstance(only_imgs[0], int):
            only_imgs = [self.images[i] for i in only_imgs]
        elif not isinstance(only_imgs[0], str):        
            logging.error("Argument 'only_imgs' must be either None, a list of images indices (List[int]) or a list of images paths (List[str])")
            return

        for img in only_imgs:
            if img not in self._ctx_masks:
                logging.warning(f"No masks in context for image {img}")
                continue
            masked = cv2.imread(img)
            masked = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
            for mask,box in zip(self._ctx_masks[img].all_masks, self._ctx_detections[img].all_boxes):
                # masked = imtools.segment_annotated_image(masked, mask, color=[30, 6, 255], alpha=0.6)
                masked = imtools.apply_image_mask(masked, binary_mask=mask, bounding_box=box)
            plt.axis("off")
            plt.imshow(masked)
            plt.show()


    def export_detections(self, only_imgs: Union[List[int], List[str], None] =  None, exp_dir="exp_analysis"):
        if only_imgs is None:
            only_imgs = self.images
        elif isinstance(only_imgs[0], int):
            only_imgs = [self.images[i] for i in only_imgs]
        elif not isinstance(only_imgs[0], str):
            logging.error("Argument 'only_imgs' must be either None, a list of images indices (List[int]) or a list of images paths (List[str])")
            return

        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir, exist_ok=True)

        for img in only_imgs:
            if img not in self._ctx_detections:
                logging.warning(f"No detections found for image {img}")
                continue
            save_name = f"det_{os.path.basename(img)}"
            imtools.save_image_detection(img, self._ctx_detections[img].all_boxes, save_name=save_name, save_dir=exp_dir)
