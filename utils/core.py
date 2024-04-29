import numpy as np
import torch
from torch import Tensor
from typing import *
from data.types import *
from data.constants import VERTEBRA_NAMES
from sklearn.impute import KNNImputer


def vertebra_to_pairs(vertebra: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """

    Args:
        vertebra: Tensor of shape (N, 2)
    """
    # A vertebra has the following points:
        # up -- um -- ua
        # lp -- lm -- la

    # Algorithm 1: Only works if the rotation is not too large
    # Sort by x and y and make pairs, assuming up and lm have the lowest x-values

    # Get indexes of the two points with smallest x
    idx = torch.argsort(vertebra[:, 0])
    posterior_idx = idx[:2]

    # Get indexes of the two points with largest x
    anterior_idx = idx[-2:]

    # Get two other points
    middle_idx = idx[2:4]

    # Sort the points
    posterior = vertebra[posterior_idx]
    middle = vertebra[middle_idx]
    anterior = vertebra[anterior_idx]

    # Sort by lowest y
    posterior = posterior[posterior[:, 1].argsort()]
    middle = middle[middle[:, 1].argsort()]
    anterior = anterior[anterior[:, 1].argsort()]

    # Safeguard
    ha = (anterior[0,:] - anterior[1,:]).norm(dim=-1)
    hp = (posterior[0,:] - posterior[1,:]).norm(dim=-1)
    hm = (middle[0,:] - middle[1,:]).norm(dim=-1)

    mpr = hm / hp
    mar = hm / ha

    # Symptoms of too large rotation

    if (mpr > 2 and mar > 2) or (mpr > 0.7 and mar < 0.55):

        
        # Get leftmost and rightmost point indices
        idx = torch.argsort(vertebra[:, 0])
        leftmost = idx[0]
        rightmost = idx[-1]

        # Sort remaining points by y
        remaining = torch.tensor([i for i in range(6) if i not in [leftmost, rightmost]])
        remaining = vertebra[remaining]
        remaining = remaining[remaining[:, 1].argsort()]

        # Get two top and bottom points
        top = remaining[-2:]
        bottom = remaining[:2]

        # Sort by x
        top = top[top[:, 0].argsort()]
        bottom = bottom[bottom[:, 0].argsort()]

        left_top = top[0]
        right_top = top[1]
        left_bottom = bottom[0]
        right_bottom = bottom[1]

        leftmost = vertebra[leftmost]
        rightmost = vertebra[rightmost]

        # Get the two points with the largest x
        if left_top[1] > right_top[1]:
            
            up = left_top
            lp = leftmost
            um = right_top
            lm = left_bottom
            ua = rightmost
            la = right_bottom
        else:
            up = leftmost
            lp = left_bottom
            um = left_top
            lm = right_bottom
            ua = right_top
            la = rightmost

        posterior = torch.stack([up, lp])
        middle = torch.stack([um, lm])
        anterior = torch.stack([ua, la])


    return posterior, middle, anterior
def as_ordered(vertebra: Tensor) -> Tensor:

    posterior, middle, anterior = vertebra_to_pairs(vertebra)

    return torch.cat([posterior, middle, anterior], dim=0)

    




def normalize_keypoints(keypoints: Tensor, height: int, width: int) -> Tensor:
    """
    Normalizes the keypoints to the range [0, 1].

    Args:
        keypoints: Tensor of shape (n_vertebrae, n_keypoints, 2)
        height: Height of the image
        width: Width of the image

    Returns:
        Tensor of shape (n_vertebrae, n_keypoints, 2)
    """
    keypoints = keypoints.view(-1, 2)
    return keypoints / torch.tensor([width, height], dtype=keypoints.dtype)

def normalize_bbox(bbox: Tensor, height: int, width: int) -> Tensor:

    bbox = bbox.view(-1, 4)
    return bbox / torch.tensor([width, height, width, height], dtype=bbox.dtype)

def bbox_from_keypoints(keypoints: Tensor) -> Tensor:
    """
    Calculates the bounding box from the keypoints in the format (xyxy).

    Args:
        keypoints: Numpy array of shape (n_vertebrae, n_keypoints, 2)

    Returns:
        Numpy array of shape (n_vertebrae, 4)
    """
    min_x, _ = keypoints[:, :, 0].min(axis=1)
    max_x, _ = keypoints[:, :, 0].max(axis=1)
    min_y, _ = keypoints[:, :, 1].min(axis=1)
    max_y, _ = keypoints[:, :, 1].max(axis=1)

    return torch.stack([min_x, min_y, max_x, max_y], axis=1).to(keypoints.device)


def apply_imputation(keypoints: Tensor, imputer: KNNImputer) -> Tuple[Tensor, Tensor]:
    """
    Applies the imputation to the keypoints.

    Args:
        keypoints: Tensor of shape (n_vertebrae, n_keypoints, 2)
        imputer: Imputer

    Returns:
        Tuple of tensors of shape (n_vertebrae, n_keypoints, 2)
    """
    keypoints = keypoints.view(-1, 12)
    keypoints = torch.tensor(
        imputer.transform(keypoints), 
        dtype=keypoints.dtype).view(-1, 6, 2)

    # Create bounding boxes from keypoints
    bboxes = bbox_from_keypoints(keypoints)

    return keypoints, bboxes

def patient_to_target(
        patient: Patient, 
        bbox_expansion: float,
        bbox_jitter: float,
        random_expansion: bool = False,
        missing_weight: float = 1e-4
        ) -> Target:
    
    """
    Converts a patient to a target class.

    Args:
        patient: Patient
        bbox_expansion: Bounding box expansion

    Returns:
        Target: Target class
    """
    
    keypoints = []
    bboxes = []
    labels = []
    visual_grades = []
    morphological_grades = []
    names = []
    indices = torch.zeros((len(VERTEBRA_NAMES)), dtype=torch.bool)
    weights = torch.ones((len(VERTEBRA_NAMES)), dtype=torch.float32)

    for vertebra in patient.vertebrae:

        if len(vertebra.coordinates) > 0:
        
            # Get keypoints
            points = vertebra.coordinates.to_numpy()

            # Get bounding box
            if random_expansion:
                bbox_expansion = np.random.uniform(0.2, bbox_expansion)

            bbox = vertebra.coordinates\
                        .to_bbox()\
                        .to_expanded(bbox_expansion)\
                        .to_format("xyxy")
            
            if bbox_jitter > 0:
                # Add jitter to the bounding box
                jitter = np.random.uniform(-bbox_jitter, bbox_jitter, 4)
                bbox += jitter
            
            # Get auxiliary information
            label = int(vertebra.typ) if vertebra.typ is not None else 0
            visual_grade = int(vertebra.grad_visuell) if vertebra.grad_visuell is not None else 0
            morph_grade  = int(vertebra.grad_morf) if vertebra.grad_morf is not None else 0
            name = vertebra.name 
            present = True

        else:

            # Get keypoints
            points = np.full((6, 2), np.nan)

            # Get bounding box
            bbox = np.full((4), np.nan)

            # Get auxiliary information
            label = 0
            # visual_grade = -1
            visual_grade = 0
            # morph_grade = -1
            morph_grade = 0
            name = vertebra.name
            present = False

        # Append to lists
        keypoints.append(points)
        bboxes.append(bbox)
        labels.append(label)
        visual_grades.append(visual_grade)
        morphological_grades.append(morph_grade)
        names.append(name)
        indices[VERTEBRA_NAMES.index(name)] = present

    if len(patient.vertebrae.annotations) > 0 and len(keypoints) == 0:
        keypoints = [ann.to_list() for ann in patient.vertebrae.annotations]

    # Convert to tensors
    keypoints = torch.tensor(np.asarray(keypoints), dtype=torch.float32)
    bboxes = torch.tensor(np.asarray(bboxes), dtype=torch.float32)
    labels = torch.tensor(np.asarray(labels), dtype=torch.int64)
    visual_grades = torch.tensor(np.asarray(visual_grades), dtype=torch.int64)
    morphological_grades = torch.tensor(np.asarray(morphological_grades), dtype=torch.int64)
    weights[~indices] = missing_weight * indices.sum() / len(indices)
    
    return Target(
        keypoints=keypoints,
        boxes=bboxes,
        labels=labels,
        types=labels,
        visual_grades=visual_grades,
        morphological_grades=morphological_grades,
        names=names,
        indices=indices,
        weights=weights,
        id=patient.moid
    )


def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.4, N+4)
    # Use logarithmic scale
    # mycmap._lut[:,-1] = np.logspace(1, 0.1, N+4)
    return mycmap