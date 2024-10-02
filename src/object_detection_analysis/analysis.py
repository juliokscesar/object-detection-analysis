from typing import Union, List, Tuple
import logging
import numpy as np
import torch
import cv2
import os
import matplotlib.pyplot as plt

from scg_detection_tools.models import BaseDetectionModel, get_opt_device, from_type
from scg_detection_tools.detect import Detector
from scg_detection_tools.utils.file_handling import read_yaml, file_exists
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
    },

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
            tasks: Tuple[str, List[BaseAnalysisTask]] = None,
        ):
        # Read from yaml if string passed
        if isinstance(config, str):
            config = read_yaml(config)
        self._config = {
            key: config[key] if key in config else DEFAULT_ANALYSIS_CONFIG[key] for key in DEFAULT_ANALYSIS_CONFIG 
        }
        
        if det_model is None:
            det_model = from_type(config["model_type"], config["model_path"], config["data_classes"])
        self._detector = Detector(det_model, detection_params=config["detection_parameters"])
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
        
        self._tasks = tasks
        self._tasks_results = {name: None for (name,_) in tasks}
        # decide if should run YOLO and/or SAM2 based on tasks requirements
        self._detect = self._segment = False
        for _, task in tasks:
            if task.require_detections:
                self._detect = True
            if task.require_masks:
                self._segment = True
            if self._detect and self._segment:
                break

    @property 
    def images(self):
        return self._ctx_imgs

    def add_image(self, img: str):
        if not file_exists(img):
            raise ValueError(f"File {img!r} does not exist.")
        self._ctx_imgs.append(img)

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

    def run(self):
        if self._detect:
            # run detection and free memory by deleting variables
            self._run_detections()
            self._detect = False
        if self._segment:
            # run segmentation and free mmemory by deleting variables
            self._run_segmentations()
            self._segment = False
        for name, task in self._tasks:
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

    def _run_detections(self):
        logging.info("Started running detections with context images")
        detections = self._detector(self.images)
        for img, detection in zip(self.images, detections):
            for cls_id, box in zip(detection.class_id, detection.xyxy.astype(np.int32)):
                if cls_id > len(self._config["data_classes"]):
                    logging.warning(f"DetectionAnalysisContext._run_detections: class id '{cls_id}' in detections from image {img} not in context data classes. Skipping...")
                    continue
                cls_label = self._config["data_classes"][cls_id]
                self._ctx_detections[img].class_boxes[cls_label].append(box)
                self._ctx_detections[img].all_boxes.append(box)
            logging.info(f"Finished registering detections from image {img}. Total detections: {len(detection.xyxy)}")
        logging.info("Finished running detections")
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
            for cls in cls_detections.class_boxes:
                cls_masks = self._segmentor.segment_boxes(img, boxes=cls_detections.class_boxes[cls])
                self._ctx_masks[img].class_masks[cls] = cls_masks
                self._ctx_masks[img].all_masks.extend(cls_masks.tolist())
            self._ctx_masks[img].all_masks = np.array(self._ctx_masks[img].all_masks)
        torch.cuda.empty_cache()

    def task_result(self, task_name):
        if task_name not in self._tasks_results:
            logging.error(f"No results for task {task_name!r}")
            return None
        return self._tasks_results[task_name]

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
            for mask in self._ctx_masks[img].all_masks:
                masked = imtools.segment_annotated_image(masked, mask, color=[30, 6, 255], alpha=0.6)
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
