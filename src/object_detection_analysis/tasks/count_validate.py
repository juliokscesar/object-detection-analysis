import logging
import numpy as np

from scg_detection_tools.dataset import read_dataset_annotation

from object_detection_analysis.tasks import BaseAnalysisTask, CountAnalysisTask

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

