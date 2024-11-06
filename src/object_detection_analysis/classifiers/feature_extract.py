import numpy as np
import torch
from torchvision import transforms
import cv2
from typing import Tuple
from sklearn.preprocessing import StandardScaler

import scg_detection_tools.utils.image_tools as imtools
from object_detection_analysis.classifiers import resnet_extract_features

def _CHECK_OBJ_INPUT(objX):
    if not isinstance(objX[0], np.ndarray):
        raise TypeError("'objX' passed to preprocess function must be a list of np.ndarray RGB images")

# Preprocessing functions (to be able to call clf.predict(imgs) instead of having to extract features first and then calling clf.predict(features))
# -> rn_feature_preprocess: use resnet feature extraction to train classificators
# -> channels_feature_preprocess: extract RGB, HSV and Gray values from a 32x32 image as features
def rn18_feature_preprocess(objX):
    """
    Extract 512-dimensional vector of features from ResNet34
    """
    if not isinstance(objX[0], np.ndarray):
        raise TypeError("'objX' passed to preprocess function must be a list of np.ndarray RGB images")

    from object_detection_analysis.classifiers import resnet_extract_features
    processed = []
    for obj in objX:
        processed.append(resnet_extract_features(obj, resnet=18))
    return np.array(processed)

def rn34_feature_preprocess(objX):
    """
    Extract 512-dimensional vector of features from ResNet34
    """
    if not isinstance(objX[0], np.ndarray):
        raise TypeError("'objX' passed to preprocess function must be a list of np.ndarray RGB images")

    from object_detection_analysis.classifiers import resnet_extract_features
    processed = []
    for obj in objX:
        processed.append(resnet_extract_features(obj, resnet=34))
    return np.array(processed)

def rn50_feature_preprocess(objX):
    """
    Extract 512-dimensional vector of features from ResNet50
    """
    if not isinstance(objX[0], np.ndarray):
        raise TypeError("'objX' passed to preprocess function must be a list of np.ndarray RGB images")

    from object_detection_analysis.classifiers import resnet_extract_features
    processed = []
    for obj in objX:
        processed.append(resnet_extract_features(obj, resnet=50))
    return np.array(processed)

def resnet_feature_preprocess(objX, version: int, cspace="RGB"):
    """
    Extract vector of features from image using ResNet
    """
    _CHECK_OBJ_INPUT(objX)

    processed = []
    for obj in objX:
        cvt = imtools.convert(obj, cspace)
        processed.append(resnet_extract_features(cvt, resnet=version))

    return np.array(processed)

def flatten_feature_preprocess(objX: np.ndarray, stdhw: Tuple[int,int] = (32,32), cspace: str = "RGB") -> np.ndarray:
    """
    Resize image to stdhw and return flattened np.ndarray
    """
    _CHECK_OBJ_INPUT(objX)
    normalizer = StandardScaler()

    processed = []
    for obj in objX:
        proc = cv2.resize(obj, stdhw, interpolation=cv2.INTER_CUBIC)
        if cspace != "RGB":
            proc = imtools.convert(proc, cspace)
        processed.append(proc.flatten())
    processed = np.array(processed)
    return normalizer.fit_transform(processed).astype(np.float32)

def channels_feature_preprocess(objX, stdhw: Tuple[int,int] = (32,32)):
    """
    Extract RGB, HSV and Gray channels from objects.
    """
    _CHECK_OBJ_INPUT(objX)
    
    processed = []
    for obj in objX:
        rgb = cv2.resize(obj, stdhw)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        attributes = np.concatenate((rgb.flatten(), hsv.flatten(), gray.flatten()))
        processed.append(attributes)

    return np.array(processed)

def norm_image_to_tensor(objX: np.ndarray, stdhw: Tuple[int,int] = (32,32), cspace: str = "RGB"):
    """
    Take all images from input, transform to torch.tensor and normalize them
    """
    _CHECK_OBJ_INPUT(objX)

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(stdhw),
        transforms.Normalize(MEAN, STD),
    ])
    trans = []
    for obj in objX:
        obj = imtools.convert(obj, cspace)
        if obj.ndim == 2: 
            obj = cv2.cvtColor(obj, cv2.COLOR_GRAY2RGB)
        trans.append(transform(obj))
    return torch.stack(trans)