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
from models.backbones.DINO.util.misc import nested_tensor_from_tensor_list, NestedTensor


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

            elif isinstance(value, list | tuple):
                new = []
                try:
                    for item in value:
                        new.append(item.to(device)) 

                except Exception as e:
                    print(f"Error in {name}: {e}")
                    raise e

                if isinstance(value, tuple):
                    setattr(self, name, tuple(new))
                else:
                    setattr(self, name, new)

            elif isinstance(value, dict):
                new = {}
                for k, v in value.items():
                    new[k] = v.to(device)
                
                setattr(self, name, new)
                
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
        compression (Tensor): The visual grades (deformation degree) of the vertebrae (N)
        morphology (Tensor): The morphological grades (shapes) of the vertebrae (N)
        labels (Tensor): The labels of the vertebrae (N)
        indicator (Tensor): Indicates presence or absence of vertebrae (N)
        weights (Tensor): Frequency labels for the labels (N)
        id (Optional[str]): The patient id
    """

    keypoints: Tensor
    boxes: Tensor
    compression: Tensor
    morphology: Tensor
    labelling: Literal["compression", "morphology", "ones", "zeros"]
    indicator: Tensor
    weights: Tensor
    names: Optional[List[str] | str] = None
    id: Optional[str] = None

    @property
    def labels(self) -> Tensor:
        """Return the labels of the targets."""
        
        if self.labelling == "compression":
            return self.compression
        elif self.labelling == "morphology":
            return self.morphology
        elif self.labelling == "ones":
            return torch.ones_like(self.indicator, dtype=torch.long)
        elif self.labelling == "zeros":
            return torch.zeros_like(self.indicator, dtype=torch.long)
        else:
            return torch.ones_like(self.indicator, dtype=torch.long)

        
    def to_dict(self, not_nan: bool = False):
        """Return a dictionary with the elements of the target."""
        if not_nan:
            d = {
                "keypoints": self.keypoints[self.indicator],
                "boxes": self.boxes[self.indicator],
                "compression": self.compression[self.indicator],
                "morphology": self.morphology[self.indicator],
                "labels": self.labels[self.indicator],
                "weights": self.weights[self.indicator],
                "names": [self.names[i] for i in range(len(self.names)) if self.indicator[i]],
                "indicator": self.indicator,
                "id": self.id
            }
            return d
        else:
            return asdict(self)

    
@dataclass
class Batch(BaseType):
    """
    Batch structure containing an image and a list of targets corresponding to vertebrae.

    Args:
        x (Tensor): The image tensor (C,H,W)
        y (List[Target]): The list of targets
        original_sizes (Optional[List[Tuple[int, int]]]): The original sizes of the images as inputed in the model
    """
    images: NestedTensor
    targets: Tuple[Target]
    original_sizes: Optional[Tuple[Tuple[int, int]]] = None

    @property
    def keypoints(self) -> Tensor:
        """Return the keypoints of the targets."""
        return torch.cat([target.keypoints for target in self.targets])
    
    @property
    def boxes(self) -> Tensor:
        """Return the bounding boxes of the targets."""
        return torch.cat([target.boxes for target in self.targets])
    
    @property
    def compressions(self) -> Tensor:
        """Return the visual grades of the targets."""
        return torch.cat([target.compression for target in self.targets])
    
    @property
    def morphologies(self) -> Tensor:
        """Return the morphological grades of the targets."""
        return torch.cat([target.morphology for target in self.targets])
    
    @property
    def indicators(self) -> Tensor:
        """Return the indicators of the targets."""
        return torch.cat([target.indicator for target in self.targets])
    
    @property
    def ids(self) -> List[str]:
        """Return the patient ids of the targets."""
        return [target.id for target in self.targets]
    
    def targets_to_records(self, not_nan: bool = False) -> List[Dict[str, Any]]:
        """Return a list of dictionaries with the targets."""
        return [{**target.to_dict(not_nan), "labels": target.labels} for target in self.targets]


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

    compression: Tensor
    morphology: Tensor

@dataclass
class VertebraPrediction(BaseType):

    keypoints: PointPrediction
    compression: Tensor
    morphology: Tensor
    image_logits: ClassPrediction
    keypoint_logits: ClassPrediction

   
@dataclass
class VertebraOutput(BaseType):
    keypoints: Optional[PointPrediction] = None
    compression: Optional[Tensor] = None
    morphology: Optional[Tensor] = None

@dataclass
class VertebraModelOutput(BaseType):
    
    loss: Tensor
    images: Tensor
    true: VertebraOutput
    prediction: VertebraOutput