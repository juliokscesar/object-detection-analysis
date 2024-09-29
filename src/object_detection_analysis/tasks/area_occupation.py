from typing import List, Union

from object_detection_analysis.tasks import BaseAnalysisTask

class ObjectAreaOccupationTask(BaseAnalysisTask):
    def __init__(self, image_order: Union[List[int], None] = None,  export_csv=False, plot=True):
        super().__init__()
        self._require_detections = True
        self._require_masks = True

        self._config = {
            "image_order": image_order,
            "plot": plot,
            "export_csv": export_csv,
        }
    
    def run(self):
        if (self._ctx_detections is None) or (self._ctx_masks is None):
            pass
