import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from object_detection_analysis.tasks import BaseAnalysisTask
from object_detection_analysis.ctx_data import ContextDetectionBoxData

class CountAnalysisTask(BaseAnalysisTask):
    def __init__(self, plot=False, **plot_ax_kwargs):
        super().__init__()
        self._require_detections = True
        self._require_masks = False
        self._can_plot = True

        # self._config = {
        #     "export_csv": export_csv,
        #     "plot_all": plot_all,
        #     "plot_per_class": plot_per_class,
        #     "save_plots": save_plots,
        # }
        self._config = {
            "plot": plot,
            "plot_ax_kwargs": plot_ax_kwargs,
        }
            
    def run(self):
        # Check if any data missing
        if (self._ctx_imgs is None) or (self._ctx_detections is None):
            logging.error("Trying to run CountAnalysisTask but either context images or detections weren't set.")
            return None

        result = {
            img: self._count_per_cls(self._ctx_detections[img]) for img in self._ctx_detections
        }
        df = self.result_dataframe(result)

        # if self._config["export_csv"]:
        #     df.to_csv("count_analysis_task_out.csv")

        # fig_plots = []
        # if self._config["plot_all"]:
        #     fig_all, ax = plt.subplots(layout="tight")
        #     ax.plot(df["img_idx"], df["all"], marker='o')
        #     ax.set(xlabel="Image ID", ylabel="Object count")
        #     fig_plots.append(("plot_all", fig_all))

        # if self._config["plot_per_class"]:
        #     fig_pcls, ax = plt.subplots(layout="tight")
        #     for cls in df.columns[1:]:
        #         if cls == "all":
        #             continue
        #         ax.plot(df["img_idx"], df[cls], marker='o', label=cls)
        #     ax.legend()
        #     ax.set(xlabel="Image ID", ylabel="Object count")
        #     fig_plots.append(("plot_per_class", fig_pcls))

        # if self._config["save_plots"]:
        #     for name, fig in fig_plots:
        #         fig.savefig(f"analysis_exp/plots/count_{name}.png")
        # plt.show()

        return df

    @staticmethod
    def _count_per_cls(img_detections: ContextDetectionBoxData):
        results = {"total": len(img_detections.all_boxes)}
        for cls in img_detections.object_classes:
            if cls not in results:
                results[cls] = len(img_detections.class_boxes[cls])
            else:
                results[cls] += len(img_detections.class_boxes[cls])
        return results

    