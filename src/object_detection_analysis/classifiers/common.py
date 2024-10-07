import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from object_detection_analysis.classifiers import KNNClassifier, SVMClassifier, MLPClassifier, CNNFCClassifier

CLASSIFIERS_MODELS = {
    "knn": (KNNClassifier, "knn_k4.skl"),
    "resnet18_knn": (KNNClassifier, "knn_rn18_k6.skl"),

    "svm": (SVMClassifier, "svm.skl"),
    "resnet18_svm": (SVMClassifier, "svm_rn18.skl"),
    
    "mlp": (MLPClassifier, "mlp.pt"),
    "resnet18_mlp": (MLPClassifier, "mlp_rn18.pt"),
    "resnet34_mlp": (MLPClassifier, "mlp_rn34.pt"),
    "resnet50_mlp": (MLPClassifier, "mlp_rn50.pt"),

    "cnn_fc": (CNNFCClassifier, "cnn.pt"),
}
def classifier_from_name(name: str, ckpt_path: str = None, to_optimal_device=True):
    if name not in CLASSIFIERS_MODELS:
        logging.fatal(f"Classifer name {name!r} is invalid")
        return None
    clf_class, clf_file = CLASSIFIERS_MODELS[name]
    if ckpt_path is not None:
        clf_file = ckpt_path
    clf = clf_class.from_state(clf_file)
    if (to_optimal_device):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clf.to(device)
    return clf
 

######################################################################
g_RESNET_INSTANCES = { 
    18: None,
    34: None,
    50: None,
}
g_RESNET_PREPROCESS = None
### Function to extract feature from images using ResNet
def resnet_extract_features(img: np.ndarray, resnet: int = 18):
    if resnet not in g_RESNET_INSTANCES:
        raise ValueError("'resnet' must be either 18, 34 or 50")
    if (g_RESNET_INSTANCES[resnet] is None) or (g_RESNET_PREPROCESS is None):
        _init_resnet(resnet)
    
    proc = g_RESNET_PREPROCESS(img).unsqueeze(0)
    with torch.no_grad():
        features = g_RESNET_INSTANCES[resnet](proc)
    return features.squeeze().cpu().numpy(force=True)

def _init_resnet(which: int = 18):
    global g_RESNET_INSTANCES, g_RESNET_PREPROCESS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if which == 18:
        g_RESNET_INSTANCES[which] = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif which == 34:
        g_RESNET_INSTANCES[which] = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    elif which == 50:
        g_RESNET_INSTANCES[which] = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    g_RESNET_INSTANCES[which] = nn.Sequential(*list(g_RESNET_INSTANCES[which].children())[:-1])
    for p in g_RESNET_INSTANCES[which].parameters():
        p.requires_grad = False

    g_RESNET_INSTANCES[which].to(device)
    g_RESNET_INSTANCES[which].eval()

    if g_RESNET_PREPROCESS is None:
        g_RESNET_PREPROCESS = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.to(device)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
