from typing import *

from utils.types import Batch, Output, Target
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from models.backbones.DINO.util.misc import NestedTensor
import timm
import sklearn.metrics
from utils.evaluate import *
import kornia.augmentation as K
from kornia.geometry.transform import translate
import matplotlib.pyplot as plt

from models.vertebra.models import SingleVertebraClassifier
from sklearn.ensemble import RandomForestClassifier


class SpineImageClassifier(L.LightningModule):

    def __init__(self, grade_weights: List[float], type_weights: List[float]) -> None:
        super().__init__()

        n_grades = 4
        n_types = 3

        self.grade_weights = torch.tensor(grade_weights)
        self.type_weights = torch.tensor(type_weights)

        # Use a pretrained inception resnet v2 model
        base = timm.create_model('inception_resnet_v2', pretrained=True)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        # self.encoder = torch.compile(self.encoder, mode="reduce-overhead")

        # Grade head
        self.grade_head = nn.Linear(1536, n_grades)

        # Type head
        # self.type_head = nn.Linear(1536, n_types)

        self.augmentation = K.AugmentationSequential(
            K.RandomAffine(
                degrees=15,
                translate=(0.10, 0.10),
                shear=20,
                scale=(0.9, 1.1)
            ),
            K.RandomBrightness((0.9, 1.1))
        )

        self.trues = []
        self.preds = []

    def forward(self, x: NestedTensor) -> Dict[str, Tensor]:
        
        x = x.tensors

        # Translate to middle
        # height, width = x.shape[-2:]
        # x = translate(x, 0, -height//2, 0, -width//2)
        x = self.augmentation(x)

        z = self.encoder(x)

        y_grades = self.grade_head(z)
        # y_types = self.type_head(z)

        return {
            "pred_grades": y_grades,
            # "pred_types": y_types
        }
    
    def step(self, batch: Batch, batch_idx: int, name: str = "", **kwargs) -> Tuple[Tensor, Dict[str, Tensor], Output]:
            
        x, y = batch.x, batch.y
        batch_size = x.tensors.shape[0]
    
        output = self(x)

        grades, types = self.preprocess_targets(y)

        self.grade_weights = self.grade_weights.to(grades.device)
        self.type_weights = self.type_weights.to(types.device)

        # Losses
        loss_grades = F.cross_entropy(output["pred_grades"], grades, weight=self.grade_weights)

        loss = loss_grades #+ loss_types
    
        # Logging
        self.log(f"{name}/loss", loss, batch_size=batch_size, **kwargs)
    
        return loss, {"grades": grades, "types": types}, output
    
    def preprocess_targets(self, targets: List[Target]) -> Tuple[Tensor, Tensor]:
        
        grades = torch.stack([t.visual_grades[t.indices] for t in targets], dim=0)
        types = torch.stack([t.labels[t.indices] for t in targets], dim=0)

        # Only take the maxium label for each image
        grades = grades.max(dim=1).values
        types = types.max(dim=1).values

        return grades, types
    
    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:

        loss, y, output = self.step(batch, batch_idx, "train_stage", on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:

        loss, y, output = self.step(batch, batch_idx, "val_stage", on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.trues.append(y)
        self.preds.append(output)

        return loss
    
    def test_step(self, batch: Batch, batch_idx: int) -> Tensor:

        loss, y, output = self.step(batch, batch_idx, "test_stage", on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.trues.append(y)
        self.preds.append(output)

        return loss
    
    def on_validation_epoch_end(self) -> None:

        self.on_any_test_end("val_stage")

        self.trues = []
        self.preds = []

    def on_test_epoch_end(self) -> None:

        self.on_any_test_end("test_stage")

        self.trues = []
        self.preds = []
    
    def on_any_test_end(self, name: str) -> None:

        grade_trues = torch.cat([t["grades"] for t in self.trues], dim=0).cpu().numpy()
        grade_preds = torch.cat([p["pred_grades"] for p in self.preds], dim=0).cpu().numpy()

        # type_trues = torch.cat([t["types"] for t in self.trues], dim=0).cpu().numpy()
        # type_preds = torch.cat([p["pred_types"] for p in self.preds], dim=0).cpu().numpy()

        self.trues = []
        self.preds = []

        grade_groups =[
                ("normal", ([0, ], [1, 2, 3])),
                ("mild",([1, ], [0, 2, 3])),
                ("moderate",([2, ], [0, 1, 3])),
                ("severe",([3, ], [0, 1, 2])),
                ("normal+mild",([0, 1], [2, 3])),
            ]
        type_groups = [
                ("normal", ([0, ], [1, 2])),
                ("wedge", ([1, ], [0, 2])),
                ("concave", ([2, ], [0, 1])),
            ]
        
        for target, all_groups, trues, preds in [("grades", grade_groups, grade_trues, grade_preds),
                                                #   ("types", type_groups, type_trues, type_preds)
                                                  ]:
        
            for group_name, groups in all_groups:
                # Compute ROC curve for a multi-class classification problem using the One-vs-Rest (OvR) strategy
                trues_binary, preds_grouped = grouped_classes(trues, preds, groups, n_classes=preds.shape[-1])

                roc = grouped_roc_ovr(trues, preds, groups, n_classes=preds.shape[-1])
                
                # Compute relevant metrics
                auc     = roc["roc_auc"]
                youden  = roc["youden_threshold"]
                preds_thresh   = (preds_grouped > youden).astype(int)

                self.log(f"{name}/{target}/{group_name}/auc", auc, prog_bar=False, on_epoch=True, on_step=False)

                try:
                    # Compute confusion matrix
                    cm = sklearn.metrics.confusion_matrix(trues_binary, preds_thresh, labels=[0,1])

                    # Compute metrics
                    # Sensitivity, specificity, precision, f1-score
                    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
                    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
                    precision   = cm[1, 1] / (cm[1, 1] + cm[0, 1])
                    accuracy    = (cm[0, 0] + cm[1, 1]) / cm.sum()

                    f1_score    = 2 * (precision * sensitivity) / (precision + sensitivity)

                    prevalence = trues_binary.sum()  


                    # Log metrics
                    self.log(f"{name}/{target}/{group_name}/youden", youden, prog_bar=False, on_epoch=True, on_step=False)
                    self.log(f"{name}/{target}/{group_name}/sensitivity", sensitivity, prog_bar=False, on_epoch=True, on_step=False)
                    self.log(f"{name}/{target}/{group_name}/specificity", specificity, prog_bar=False, on_epoch=True, on_step=False)
                    self.log(f"{name}/{target}/{group_name}/precision", precision, prog_bar=False, on_epoch=True, on_step=False)
                    self.log(f"{name}/{target}/{group_name}/accuracy", accuracy, prog_bar=False, on_epoch=True, on_step=False)
                    self.log(f"{name}/{target}/{group_name}/f1_score", f1_score, prog_bar=False, on_epoch=True, on_step=False)
                    self.log(f"{name}/{target}/{group_name}/prevalence", prevalence, prog_bar=False, on_epoch=True, on_step=False)

                except Exception:
                    pass
