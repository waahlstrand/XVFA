from typing import *
from data.types import Batch, Output
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
import lightning as L
from models.vertebra.models import SingleVertebraClassifier
from utils.evaluate import *

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Detector(L.LightningModule):

    def __init__(self, 
                lr: float = 1e-4,
                lr_backbone: float = 1e-5,
                weight_decay: float = 1e-4,
                batch_size: int = 8,
                vertebra_classifier_path: Optional[str] = None,
                 ) -> None:
        
        super().__init__()

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.vertebra_classifier_path = vertebra_classifier_path

        if self.vertebra_classifier_path is not None:
            self.vertebra_classifier = SingleVertebraClassifier.load_from_checkpoint(self.vertebra_classifier_path)
            self.vertebra_classifier.eval()

        self.metrics  = nn.ModuleDict({
            name: self.build_metrics() for name in ["train_stage", "val_stage", "test_stage"]
        })

        self.test_true = []
        self.test_pred = []


    def forward(self, x: Tensor) -> Output:
        raise NotImplementedError
    
    def __call__(self, *args: Any, **kwds: Any) -> Output:
        return super().__call__(*args, **kwds)
    
    def step(self, batch: Batch, batch_idx: int, name: str = "", **kwargs) -> Tuple[Tensor, Dict[str, Tensor]]:
        raise NotImplementedError
    
    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:

        loss, output = self.step(batch, batch_idx, name="train_stage", batch_size=self.batch_size, prog_bar=False, on_step=True, on_epoch=True)
        
        return {
            "loss": loss,
            **output
            # **{f"pred_{k}": v for k, v in output.items()},
        }
    
    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:

       loss, output = self.step(batch, batch_idx, name="val_stage", batch_size=self.batch_size, prog_bar=False, on_step=False, on_epoch=True)
       
       return {
            "loss": loss,
            **output
            # **{f"pred_{k}": v for k, v in output.items()},
        }
    
    def on_validation_epoch_end(self) -> None:
        pass

    
    def test_step(self, batch: Batch, batch_idx: int) -> Tensor:
        
        loss, output = self.step(batch, batch_idx, name="test_stage", batch_size=self.batch_size, prog_bar=False, on_step=False, on_epoch=True)

        x, y = batch.x, batch.y
        sizes  = batch.original_sizes

        # Format the output to be compatible with the postprocessor
        #    output = {f"pred_{k}": v for k, v in output.items()}
        processed = self.postprocessors["bbox"](output, torch.tensor(sizes, device=output["pred_boxes"].device))
        boxes  = processed["pred_boxes"]

        # Get cropped bounding boxes from detector
        cropped = roi_align(x, boxes, output_size=(224, 224))

        # Get the predictions from the vertebra classifier
        vertebra = self.vertebra_classifier(cropped)
        # image_type_logits    = vertebra.type_logits
        image_grade_logits   = vertebra.grade_logits

        keypoints       = keypoints.reshape(*vertebra.keypoints.mu.shape)

        # Get classification from keypoints
        keypoint_logits         = self.vertebra_classifier.classifier(vertebra.keypoints.mu)
        # keypoint_type_logits    = keypoint_logits.type_logits
        keypoint_grade_logits   = keypoint_logits.grade_logits

        grades_pred = self.vertebra_classifier.prediction(keypoint_grade_logits, image_grade_logits)
        # types_pred  = self.vertebra_classifier.prediction(keypoint_type_logits, image_type_logits)

        # Format the target to be compatible with the postprocessor
        grades = [target.visual_grades for target in y]
        

        self.test_true.extend(grades)
        self.test_pred.append(grades_pred)

        return {
                "loss": loss,
                **output
            }
    
    def on_test_epoch_end(self) -> None:

        self.test_pred = torch.cat(self.test_pred, dim=0)
        self.test_true = torch.cat(self.test_true, dim=0)

        # Compute classification metrics
        for groups in (("normal+mild", ([0, 1], [2, 3]))):
            metrics = classification_metrics(self.test_true, self.test_pred, all_groups=[groups])
            for metric in metrics:
                # self.log_dict(metric)
                print(metric)

        
        
        # mAP = self.metrics["test_stage"]["map"](self.test_pred, self.test_true)

        # self.log_dict({f"test_stage/{k}": v for k, v in mAP.items()})



        self.test_true = []
        self.test_pred = []

    
    def configure_optimizers(self) -> Any:
        
        param_dicts = [
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad],
                  "lr": self.lr,
              },
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]
        optimizer   = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

        return {
            "optimizer": optimizer,
        }
    
    def build_metrics(self) -> nn.ModuleDict:
            
            raise NotImplementedError