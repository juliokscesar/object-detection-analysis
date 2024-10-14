import logging
import numpy as np

from scg_detection_tools.dataset import read_dataset_annotation
from scg_detection_tools.utils.file_handling import get_annotation_files

from object_detection_analysis.tasks import BaseAnalysisTask, CountAnalysisTask

# TODO: Get dataset path as input and output training, validate and test metrics
# -> also have options to choose modes
class CountValidateModelTask(BaseAnalysisTask):
    def __init__(self, annotations_path: str, export_csv = False):
        super().__init__()
        self._config = {
            "export_csv": export_csv,
            "annotations_path": annotations_path,
        }
        self._require_detections = True
        self._require_masks = False
        self._can_plot = False

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
                true_count.append(len(ann))
        true_count = np.array(true_count)
        pred_count = [
            CountAnalysisTask._count_per_cls(self._ctx_detections[img])["all"] if img in self._ctx_detections else 0 for img in self._ctx_imgs
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
        
        errors = pred_count - true_count
        relative = np.divide(
            errors.astype(np.float64),
            true_count.astype(np.float64),
            out=np.zeros_like(errors.astype(np.float64)),
            where=true_count != 0,
        )
        me = errors.mean()
        mwe = np.sum(errors * true_count)/true_count.sum()
        mae = np.abs(errors).mean()
        mwae = np.sum(np.abs(errors) * true_count)/true_count.sum()
        mare = np.abs(relative).mean()
        mse = (errors**2).mean()
        rmse = np.sqrt(mse)
        rmsre = np.sqrt((relative**2).mean())
        rrmse = np.sqrt((errors**2).mean() / (true_count**2).sum())
        stderror = np.sqrt( ((errors - errors.mean())**2.0).mean() )
        return {
            "me": me, "mwe": mwe, "mae": mae, "mwae": mwae, "mse": mse, "rmse": rmse, "stderror": stderror,
            "mare": mare, "rmsre": rmsre, "rrmse": rrmse,
        }

    def _load_annotations(self):
        ann_files = get_annotation_files(self._ctx_imgs, self._config["annotations_path"])
        self._annotation_files = ann_files
        return (self._annotation_files is not None)
