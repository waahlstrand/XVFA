import sklearn.metrics
import sklearn.preprocessing
import numpy as np
from typing import *
from pathlib import Path
import pandas as pd
from torch import Tensor
import torch
from models.vertebra.classifiers import VertebraParameters


def detach_dict(d: Dict[str, Tensor]) -> Dict[str, Tensor]:
    out = {}
    for k, v in d.items():
        if isinstance(v, Tensor):
            out[k] = v.detach()
        else:
            out[k] = v

    return out

def dict_to_device(d: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    out = {}
    for k, v in d.items():
        if isinstance(v, Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v

    return out

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.4, N+4)
    # Use logarithmic scale
    # mycmap._lut[:,-1] = np.logspace(1, 0.1, N+4)
    return mycmap

def grouped_classes(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: Tuple[List[int], List[int]],
        n_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Group the true and predicted labels according to the provided groups.

    Args:
        y_true: True labels (n_samples)
        y_pred: Predicted labels (n_samples)
        groups: Tuple of lists of class indices e.g

    Returns:
        Tuple of grouped true and predicted labels
    """
    assert len(groups) == 2, "groups must be a tuple of two lists"
    assert len(groups[0]) + len(groups[1]) == n_classes, "The sum of the lengths of the two lists in groups must be equal to n_classes"
    assert set(groups[0] + groups[1]) == set(range(n_classes)), "The union of the two lists in groups must be equal to the set of all class indices"
    
    # Binarize the true labels for each group. If the 
    # true label is in the first group, the binarized label is 1,
    # otherwise it is 0.
    y_true_binary = np.zeros((y_true.shape[0]))
    for i, group in enumerate(groups):
        y_true_binary = np.where(np.isin(y_true, group), i, y_true_binary)

    # Sum the predicted probabilities for each group
    y_pred_grouped = np.zeros((y_pred.shape[0], 2))

    for i, group in enumerate(groups):
        y_pred_grouped[:, i] = y_pred[:, group].sum(axis=1)

    y_pred_grouped = 1-y_pred_grouped[:,0]

    return y_true_binary, y_pred_grouped

def grouped_roc_ovr(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: Tuple[List[int], List[int]],
        n_classes: int,
) -> Dict[Literal["fpr", "tpr", "thresholds", "roc_auc", "youden_threshold"], np.ndarray]:
    
    """
    Compute ROC curve for a multi-class classification problem using the One-vs-Rest (OvR) strategy,
    grouping the classes according to the provided groups.

    Args:
        y_true: True labels (n_samples)
        y_pred: Predicted labels (n_samples, n_classes)
        groups: Tuple of lists of class indices e.g. ([0, 1], [2, 3])
        n_classes: Number of classes
        label_dict: Dictionary mapping class indices to class names
    """
    y_true_binary, y_pred_grouped = grouped_classes(y_true, y_pred, groups, n_classes)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true_binary, y_pred_grouped)
    auc = sklearn.metrics.auc(fpr, tpr)

    youden_idx = np.argmax(tpr - fpr)
    youden_threshold = thresholds[youden_idx]


    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "roc_auc": auc,
        "youden_threshold": youden_threshold
    }

def roc_ovr(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        n_classes: int, 
        label_dict: Dict[int, str],
        ) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute ROC curve for a multi-class classification problem using the One-vs-Rest (OvR) strategy.

    Args:
        y_true: True labels (n_samples)
        y_pred: Predicted labels (n_samples, n_classes)
        n_classes: Number of classes
        label_dict: Dictionary mapping class indices to class names
        path: Path to save the ROC curve plot

    Returns:

    """

    lb = sklearn.preprocessing.LabelBinarizer()
    y_true_binary = lb.fit_transform(y_true)

    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = sklearn.metrics.roc_curve(y_true_binary[:, i], y_pred[:, i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

    return {
        label_dict[i]: {
            "fpr": fpr[i],
            "tpr": tpr[i],
            "thresholds": thresholds[i],
            "roc_auc": roc_auc[i],
        }
        for i in range(n_classes)
    }

def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels (n_samples,)
        y_pred: Predicted labels (n_samples,)
        label_dict: Dictionary mapping class indices to class names

    Returns:
        np.ndarray: Confusion matrix
    """
    return sklearn.metrics.confusion_matrix(y_true, y_pred, **kwargs)


def compute_classification_metrics(trues: Tensor, preds: Tensor, all_groups: List[Tuple[str, Tuple[List[int], List[int]]]], ids: Optional[List[str]] = None) -> Generator[Dict[str, float], None, None]:

    trues = trues.squeeze().cpu().numpy()
    preds = preds.cpu().numpy()
    
    if ids is not None:
        assert len(ids) == trues.shape[0], "The length of ids must be equal to the number of samples in the dataset"
        
    for group_name, groups in all_groups:

        d = {}

        grouped = grouped_metrics(trues, preds, groups)
        trues_grouped = grouped.pop("trues")
        preds_grouped = grouped.pop("preds")
        roc = grouped.pop("roc")

        preds_thresh = (preds_grouped > roc["youden_threshold"]).astype(int)

        metrics = classification_metrics(trues_grouped, preds_thresh)


        if ids is not None:
            df = pd.DataFrame({
                "id": ids,
                "true": trues_grouped,
                "pred": preds_thresh,
            })

            print(df)
            
            df = df.groupby("id").agg({"true": "max", "pred": "max"}).reset_index()

            patient_metrics = classification_metrics(df["true"].values, df["pred"].values)

            d = {
                **d,
                **{f"patient_{k}": v for k, v in patient_metrics.items()},
            }


        d = {
            **d,
            "auc": roc["roc_auc"],
            **metrics,
        }

        yield d

def grouped_metrics(trues: np.ndarray, preds: np.ndarray, groups: Tuple[str, Tuple[List[int], List[int]]]) -> np.ndarray:
    """
    Group the true and predicted labels according to the provided groups.

    Args:
        y_true: True labels (n_samples)
        y_pred: Predicted labels (n_samples)
        groups: Tuple of lists of class indices e.g

    Returns:

    """
    trues_binary, preds_grouped = grouped_classes(trues, preds, groups, n_classes=preds.shape[-1])

    roc = grouped_roc_ovr(trues, preds, groups, n_classes=preds.shape[-1])

    return {
        "trues": trues_binary,
        "preds": preds_grouped,
        "roc": roc
    }


def classification_metrics(trues: np.ndarray, preds: np.ndarray) -> Dict[str, float]:

    # Compute confusion matrix
    cm = sklearn.metrics.confusion_matrix(trues, preds, labels=[0,1])

    # Compute metrics
    # Sensitivity, specificity, precision, f1-score
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    precision   = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    accuracy    = (cm[0, 0] + cm[1, 1]) / cm.sum()

    # Get the prevalence of the positive class
    prevalence = trues.sum()  

    f1_score    = 2 * (precision * sensitivity) / (precision + sensitivity)

    d = {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "accuracy": accuracy,
        "prevalence": prevalence,
        "f1_score": f1_score,

    }

    return d



def sample_model_likelihood(model, image, n_samples=1000) -> Tuple[Tensor, Tensor]:

    likelihood, xx, yy = model.get_likelihood(image)
    likelihood = likelihood.cpu().numpy()
    xx = xx.cpu().numpy()
    yy = yy.cpu().numpy()

    X, Y = [], []

    # Loop over keypoints
    for i in range(likelihood.shape[1]):
        l = likelihood[0, i, :, :]
        sample = np.random.choice(
            a = np.arange(0, len(l.flatten())), 
            size = n_samples, 
            p = l.flatten(), 
            replace=True
            )
        
        sample_x_idx, sample_y_idx = np.unravel_index(sample, l.shape)
        sample_x, sample_y = xx[sample_x_idx, sample_y_idx], yy[sample_x_idx, sample_y_idx]

        X.append(sample_x)
        Y.append(sample_y)

    # Concatenate into shapes (n_keypoints, n_samples)
    X = torch.cat(X, dim=0).reshape(-1, n_samples)
    Y = torch.cat(Y, dim=0).reshape(-1, n_samples)

    return X, Y

def format_keypoints_for_random_forest(keypoints: Tensor) -> np.ndarray:

    vp = VertebraParameters()
    params = vp(keypoints) # Dict[str, Tensor]

    X = torch.stack([v for k, v in params.items()], dim=1).cpu().numpy()

    return X

    

    



