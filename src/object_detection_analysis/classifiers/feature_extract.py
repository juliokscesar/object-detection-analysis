import numpy as np
import torch
from torchvision import transforms
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

def channels_feature_preprocess(objX):
    """
    Extract RGB, HSV and Gray channels from objects.
    """
    if not isinstance(objX[0], np.ndarray):
        raise TypeError("'objX' passed to preprocess function must be a list of np.ndarray RGB images")

    processed = []
    for obj in objX:
        rgb = cv2.resize(obj, (32,32))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        attributes = np.concatenate((rgb.flatten(), hsv.flatten(), gray.flatten()))
        processed.append(attributes)

    return np.array(processed)

def norm_channels_feature_preprocess(objX):
    """
    Extract RGB, HSV and Gray channels from objects, normalize each feature vector and then use PCA to reduce dimensionality to 256 features.
    """
    if not isinstance(objX[0], np.ndarray):
        raise TypeError("'objX' passed to preprocess function must be a list of np.ndarray RGB images")

    processed = []
    for obj in objX:
        rgb = cv2.resize(obj, (32,32))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        attributes = np.concatenate((rgb.flatten(), hsv.flatten(), gray.flatten()))
        processed.append(attributes)
    processed = np.array(processed)
    # mean = processed.mean(axis=0, keepdims=True)
    # std = processed.std(axis=0, keepdims=True)
    # norm = (processed - mean) / (std + 1e-7)

    norm_pca_pipe = Pipeline(
        steps = [
            ("scaler", StandardScaler()), 
        ]
    )
    norm = norm_pca_pipe.fit_transform(processed)
    return norm.astype(np.float32)


def norm_image_to_tensor(objX):
    """
    Take all images from input, transform to torch.tensor and normalize them
    """
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32,32)),
        transforms.Normalize(MEAN, STD),
    ])
    trans = []
    for obj in objX:
        trans.append(transform(obj))
    return torch.stack(trans)