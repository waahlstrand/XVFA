from typing import *

from torch.nn.modules import ModuleDict
from data.types import Batch, Output, Loss, Prediction, Target
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning as L
import torchvision.models as models
from dataclasses import asdict
from models.base import Detector
from models.backbones.DINO.util.misc import NestedTensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models import resnet50
from models.spine.adapter import DINO
from argparse import Namespace
import sklearn.metrics
from utils.evaluate import *
from models.spine.dino import build_dino

def build_model(args, class_weights: List[float] = None):

    if args.model == "spine-dino":
                    
        model = SpineDINO(args, n_classes = args.n_classes, class_weights=class_weights)

    else:
        raise NotImplementedError
    
    return model


class SpineFasterRCNN(Detector):

    pass

class SpineDINO(DINO):

    def __init__(self, **kwargs) -> None:

        super().__init__(**kwargs)

        args = Namespace(**kwargs)
        
        model, criterion, postprocessors = build_dino(args)

        self.model = model
        self.criterion = criterion
        self.postprocessors = postprocessors


    def forward(self, x: NestedTensor, y: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:

        # x = self.model(x) # Dict[str, Tensor]
        x = self.model(x, y)

        return x

    def step(self, batch: Batch, batch_idx: int, name: str = "", **kwargs) -> Tuple[Tensor, Output]:

        x, y = batch.x, batch.y
        y = [{k: v for k, v in asdict(_).items() if k in ["boxes", "keypoints", "labels", "indices", "weights"]} for _ in y]

        output = self(x, y)

        # Convert the targets to the format expected by the criterion
        # y = [{k: v for k, v in asdict(_).items() if k in ["boxes", "keypoints", "labels", "indices"]} for _ in y]

        loss_dict = self.criterion(output, y)

        # Logging
        losses = [
            "loss_ce",
            "loss_bbox",
            "loss_giou",
            "loss_polynomial",
            "class_error",
            "cardinality_error",
            # "auroc",
            # "f1_score",
            # "accuracy",
        ]
        for loss_name in losses:

            self.log(f"{name}/{loss_name}", loss_dict[loss_name], **kwargs)

        weight_dict = self.criterion.weight_dict

        total = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        self.log(f"{name}/loss", total, **kwargs)

        return total, output
    
    def build_metrics(self) -> ModuleDict:
        
        return ModuleDict({
            "map": MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        })
    
