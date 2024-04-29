import sklearn.metrics
import sklearn.preprocessing
import numpy as np
from typing import *
from pathlib import Path
import pandas as pd
from torch import Tensor
import torch
from models.vertebra.classifiers import VertebraParameters

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


def classification_metrics(trues: Tensor, preds: Tensor, all_groups: List[Tuple[str, Tuple[List[int], List[int]]]]):

    trues = trues.squeeze().cpu().numpy()
    preds = preds.cpu().numpy()
        
    for group_name, groups in all_groups:
        # Compute ROC curve for a multi-class classification problem using the One-vs-Rest (OvR) strategy
        trues_binary, preds_grouped = grouped_classes(trues, preds, groups, n_classes=preds.shape[-1])

        roc = grouped_roc_ovr(trues, preds, groups, n_classes=preds.shape[-1])
            
        # Compute relevant metrics
        auc     = roc["roc_auc"]
        youden  = roc["youden_threshold"]
        preds_thresh   = (preds_grouped > youden).astype(int)

        # Compute confusion matrix
        cm = sklearn.metrics.confusion_matrix(trues_binary, preds_thresh, labels=[0,1])

        # Compute metrics
        # Sensitivity, specificity, precision, f1-score
        sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        precision   = cm[1, 1] / (cm[1, 1] + cm[0, 1])
        accuracy    = (cm[0, 0] + cm[1, 1]) / cm.sum()

        # Get the prevalence of the positive class
        prevalence = trues_binary.sum()  

        f1_score    = 2 * (precision * sensitivity) / (precision + sensitivity)

        yield {
            "group_name": group_name,
            "auc": auc,
            "youden": youden,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "accuracy": accuracy,
            "prevalence": prevalence,
            "f1_score": f1_score
        }

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

    

    



