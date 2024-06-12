# Defines the types and structures used as input and output of the models

from pathlib import Path
import json
from typing import *

import numpy as np
from PIL import Image

from dataclasses import dataclass, asdict

from torchvision.transforms.functional import crop
from torchvision.ops import box_convert
import torch
from torch import Tensor, tensor

@dataclass
class BaseType:
    """
    Base type for all types used in the models,
    implementing the `to` method to move the type to a device.

    """

    def to(self, device: str) -> "BaseType":
        """Iterate the elements of the type and move them to the device."""

        for name, value in asdict(self).items():

            if isinstance(value, Tensor):
                setattr(self, name, value.to(device))

            elif isinstance(value, list):
                if all(isinstance(item, Tensor) for item in value):
                    setattr(self, name, [item.to(device) for item in value])
                else:
                    pass
                
            else:
                pass
                
        return self
    
    def to_dict(self) -> Dict[str, Tensor]:
        """Return a dictionary with the elements of the type."""
        return asdict(self)
    

@dataclass
class Target(BaseType):
    """
    Target/label structure for the models.

    Args:
        keypoints (Tensor): The keypoints of the vertebrae (N,2)
        boxes (Tensor): The bounding boxes of the vertebrae (N,4)
        names (Optional[List[str]]): The names of the vertebrae
        visual_grades (Tensor): The visual grades (deformation degree) of the vertebrae (N)
        morphological_grades (Tensor): The morphological grades (shapes) of the vertebrae (N)
        labels (Tensor): The labels of the vertebrae (N)
        types (Tensor): The types of the vertebrae (N)
        indices (Tensor): Indicator variable (N)
        weights (Tensor): Frequency labels for the visual_grades (N)
        id (Optional[str]): The patient id
    """

    keypoints: Tensor
    boxes: Tensor
    names: Optional[List[str] | str]
    visual_grades: Tensor
    morphological_grades: Tensor
    labels: Tensor
    types: Tensor
    indices: Tensor
    weights: Tensor
    id: Optional[str] = None

    
@dataclass
class Batch(BaseType):
    """
    Batch structure containing an image and a list of targets corresponding to vertebrae.

    Args:
        x (Tensor): The image tensor (C,H,W)
        y (List[Target]): The list of targets
        original_sizes (Optional[List[Tuple[int, int]]]): The original sizes of the images as inputed in the model
    """
    images: Tensor
    targets: List[Target]
    original_sizes: Optional[List[Tuple[int, int]]] = None

    @property
    def keypoints(self) -> Tensor:
        """Return the keypoints of the targets."""
        return torch.cat([_.keypoints for _ in self.y])
    
    @property
    def boxes(self) -> Tensor:
        """Return the bounding boxes of the targets."""
        return torch.cat([_.boxes for _ in self.y])
    
    @property
    def visual_grades(self) -> Tensor:
        """Return the visual grades of the targets."""
        return torch.cat([_.visual_grades for _ in self.y])
    
    @property
    def morphological_grades(self) -> Tensor:
        """Return the morphological grades of the targets."""
        return torch.cat([_.morphological_grades for _ in self.y])


@dataclass
class PointPrediction(BaseType):
    """
    Prediction structure for the models. Contains the predicted keypoint/bbox mean and standard deviation.

    Args:
        mu (Tensor): The predicted mean of the keypoints/bbox (N,2)
        sigma (Optional[Tensor]): The predicted standard deviation of the keypoints/bbox (N,2)
    """
    mu: Tensor
    sigma: Optional[Tensor] = None

@dataclass
class Output(BaseType):
    """
    Output structure for the models. Contains the predicted keypoints/bboxes, logits, labels and scores.

    Args:
        keypoints (Optional[Prediction]): The predicted keypoints
        bboxes (Optional[Prediction]): The predicted bounding boxes
        logits (Optional[Tensor]): The predicted logits
        labels (Optional[Tensor]): The predicted labels
        scores (Optional[Tensor]): The predicted scores
    """

    keypoints: Optional[PointPrediction] = None
    bboxes: Optional[PointPrediction] = None
    logits: Optional[Tensor] = None
    labels: Optional[Tensor] = None
    scores: Optional[Tensor] = None

@dataclass
class ClassPrediction(BaseType):

    grades: Tensor
    types: Tensor

@dataclass
class VertebraPrediction(BaseType):

    keypoints: PointPrediction
    grades: Tensor
    types: Tensor
    image_logits: ClassPrediction
    keypoint_logits: ClassPrediction

   
@dataclass
class VertebraOutput(BaseType):
    keypoints: Optional[PointPrediction] = None
    grades: Optional[Tensor] = None
    types: Optional[Tensor] = None

@dataclass
class VertebraModelOutput(BaseType):
    
    loss: Tensor
    images: Tensor
    true: VertebraOutput
    prediction: VertebraOutput