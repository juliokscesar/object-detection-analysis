import logging
from typing import Union, List
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scg_detection_tools.utils.image_tools as imtools
from object_detection_analysis.tasks import BaseAnalysisTask


class ColorSpaceAnalysisTask(BaseAnalysisTask):
    def __init__(self, cspaces: List[str], on_masks=True, only_images: Union[List[str], List[int], None] = None, plot_histograms = False):
        super().__init__()
        self._require_detections = on_masks
        self._require_masks = on_masks
        self._can_plot = False

        for cspace in cspaces:
            if cspace.strip().upper() not in cspaces:
                raise ValueError(f"cspace valid options are {', '.join(list(imtools.COLOR_SPACES_CV2.keys()))}")

        self._config = {
            "on_masks": on_masks,
            "only_images": only_images,
            "cspaces": cspaces,
            "plot_histograms": plot_histograms,
        }

    def run(self):
        images = self._ctx_imgs
        img_idx = list(range(len(images)))
        if self._config["only_images"] is not None:
            if isinstance(self._config["only_images"][0], int):
                images = [images[i] for i in self._config["only_images"]]
                img_idx = self._config["only_images"]
            else:
                images = self._config["only_images"]
                img_idx = [self._ctx_imgs.index(img) for img in images]

        results = {"img_idx": [], "color_space": [], "channel": [], "mean": [], "stddev": []}
        for idx, img in zip(img_idx, images):
            for cspace in self._config["cspaces"]:
                hists, means, stddevs = self.cspace_analysis(img, cspace)
                nchannels = len(hists)
                for c in range(nchannels):
                    results["img_idx"].append(idx)
                    results["color_space"].append(cspace)
                    results["channel"].append(c)
                    results["mean"].append(means[c])
                    results["stddev"].append(stddevs[c])

                if not self._config["plot_histograms"]:
                    continue
                _,axs = plt.subplots(ncols=2, layout="tight", figsize=(12,8))
                axs[0].imshow(imtools.convert(img, cspace))
                axs[0].axis("off")
                for c in range(nchannels):
                    axs[1].plot(hists[c], label=f"{cspace} {c}")
                axs[1].set(title=f"{cspace}", xlabel="Intensity", ylabel="Frequency")
                plt.show()

        df = pd.DataFrame(results)
        return df

    def cspace_analysis(self, img: str, cspace: str):
        cvt = imtools.convert(img, cspace)
        nchannels = cvt.shape[-1] if cvt.ndim == 3 else 1
        hists = np.zeros(shape=(nchannels,256))
        if self._config["on_masks"]:
            for box, mask in zip(self._ctx_detections[img].all_boxes, self._ctx_masks[img].all_masks):
                obj = imtools.crop_box_image(cvt, box)
                hists = hists + imtools.histogram(obj, [mask])
        else:
            hists = imtools.histogram(cvt, None)

        
        intensities = np.arange(256)
        channel_means = np.zeros(nchannels, dtype=np.float64)
        channel_stddevs = np.zeros(nchannels, dtype=np.float64)
        for c in range(nchannels):
            channel_means[c] = np.sum(intensities * hists[c]) / np.sum(hists[c])
            channel_stddevs[c] = np.sqrt(np.sum((intensities - channel_means[c])**2 * hists[c]) / np.sum(hists[c]))

        return hists, channel_means, channel_stddevs
