from typing import *
import joblib

from utils.types import Batch, Output
import torch
from torch import Tensor
from models.backbones.DINO.util import box_ops
from models.spine.adapter import DINO
from argparse import Namespace
from utils.evaluate import *
from models.spine.dino import build_dino
from models.vertebra.models import SingleVertebraClassifier
from torchvision.ops import roi_align
from utils.evaluate import *

class SpineDINO(DINO):

    def __init__(self,
                lr: float = 1e-4,
                lr_backbone: float = 1e-5,
                weight_decay: float = 1e-4,
                batch_size: int = 8,
                vertebra_classifier_path: Optional[str] = None,
                random_forest_path: Optional[str] = None,
                level: Literal["vertebra", "patient"] = "vertebra",
                 **kwargs) -> None:
        
        super().__init__(lr=lr, lr_backbone=lr_backbone, batch_size=batch_size, weight_decay=weight_decay, **kwargs)

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.vertebra_classifier_path = vertebra_classifier_path
        self.random_forest_path = random_forest_path
        self.level = level

        self.save_hyperparameters()
        
        # Build the DINO model with some minor changes to the original implementation
        args = Namespace(lr=lr, lr_backbone=lr_backbone, batch_size=batch_size, weight_decay=weight_decay, **kwargs)
        model, criterion, postprocessors = build_dino(args)

        self.model = model
        self.criterion = criterion
        self.postprocessors = postprocessors

        # Initialize the test variables
        self.true = {
            "compression": [],
            "morphology": [],
            "bbox": [],
            "keypoints": [],
            "ids": []
        }

        self.pred = {
            "compression": [],
            "morphology": [],
            "bbox": [],
            "keypoints": []
        }
        
    def forward(self, batch: Batch) -> Dict[str, Tensor]:
        
        return self.model(batch.images, targets=None)
    
    def __call__(self, *args: Any, **kwds: Any) -> Dict[str, Tensor]:
        return super().__call__(*args, **kwds)

    def step(self, batch: Batch, batch_idx: int, name: str = "", **kwargs) -> Tuple[Tensor, Output]:
        """
        Perform a single forward pass through the model and compute the loss. Used in the training, validation and test steps.
        To just perform a forward pass without computing the loss, use the predict_step method, or 
        call the model directly.
        
        Args:
            batch (Batch): The batch of data to be processed
            batch_idx (int): The index of the batch
            name (str): The name of the stage
            kwargs: Additional keyword arguments used for logging
            
        Returns:
            Tuple[Tensor, Output]: The loss and the output of the model
        
        """

        targets = batch.targets_to_records()

        # Forward pass
        output = self.model(batch.images, targets=targets)

        loss_dict = self.criterion(output, targets)

        # Compute the total loss and weight the contributions of each loss in DINO
        weight_dict = self.criterion.weight_dict

        total = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Logging the losses
        if name != "predict_stage":
            for loss_name in ["loss_ce", "loss_bbox", "loss_giou", "class_error", "cardinality_error"]:
                
                self.log(f"{name}/{loss_name}", loss_dict[loss_name], **kwargs)

            self.log(f"{name}/loss", total, **kwargs)

        return total, output
    
    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:

        loss, output = self.step(batch, batch_idx, name="train_stage", batch_size=self.batch_size, prog_bar=False, on_step=True, on_epoch=True)
        
        return {"loss": loss, **output}
    
    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:

       loss, output = self.step(batch, batch_idx, name="val_stage", batch_size=self.batch_size, prog_bar=False, on_step=False, on_epoch=True)

       true, pred = self.align_predictions(output, batch)
       
       return {"loss": loss, **output}
    
    def test_step(self, batch: Batch, batch_idx: int) -> Tensor:

        loss, outputs = self.step(batch, batch_idx, name="test_stage", batch_size=self.batch_size, prog_bar=False, on_step=False, on_epoch=True)

        true, pred = self.align_predictions(outputs, batch)
            
        return {"loss": loss, **outputs}

    def predict_step(self, batch: Batch, batch_idx: int, name: str = "", **kwargs) -> Dict[str, Tensor]:

        loss, outputs = self.step(batch, batch_idx, name="predict_stage", batch_size=self.batch_size, prog_bar=False, on_step=False, on_epoch=False)

        true, pred = self.align_predictions(outputs, batch)

        return {"true": true, "pred": pred, "batch": batch} 

    def initialize_classifiers(self) -> None:
        """
        Load the vertebra classifier and the random forest model if they are provided. Used for end-to-end testing of
        vertebra detection and classification.
        """

        if self.vertebra_classifier_path is not None:
            self.vertebra_classifier = SingleVertebraClassifier.load_from_checkpoint(self.vertebra_classifier_path, strict=False).to(self.device)
            self.vertebra_classifier.eval()
            self.vertebra_classifier.freeze()

        if self.random_forest_path is not None:
            self.random_forest = joblib.load(self.random_forest_path)

    def align_predictions(self, outputs: Output, batch: Batch) -> Tuple[Dict, Dict]:

        targets = batch.targets
        # In order to compute the correct classification by the vertebra classifier
        # we need to match the predictions to the targets
        processed   = self.postprocessors["bbox"](
            outputs, 
            torch.tensor(batch.original_sizes, device=self.device), 
            as_records=True
            )

        # Match the outputs to the targets for pairwise comparison
        # but remove any missing, imputed or non-annotated targets
        records = []
        for target, size in zip(batch.targets, batch.original_sizes):
            
            scale = torch.tensor([size[1], size[0], size[1], size[0]], device=self.device)
            bbox = target.boxes[target.indicator] 
            bbox = box_ops.box_cxcywh_to_xyxy(bbox) * scale

            records.append({
                "labels": target.labels[target.indicator],
                "boxes": bbox,
            })

        indices = self.criterion.matcher(processed, records)

        for target in targets:
            for _ in target.indicator.nonzero():
                self.true["ids"].append(target.id)

        # If we are testing the vertebra detection and classification
        # with the XVFA vertebra classifier
        pred = {
            "keypoints": [],
            "compression": [],
            "morphology": [],
            "bbox": [],
        }

        true = {
            "keypoints": [],
            "compression": [],
            "morphology": [],
            "bbox": [],
            "ids": []
        }
        if self.vertebra_classifier_path is not None:

            for image, target, size, bboxes, idx in zip(batch.images, 
                                                        batch.targets, 
                                                        batch.original_sizes, 
                                                        processed["pred_boxes"], 
                                                        indices):
                
                cropped = roi_align(image.unsqueeze(0), 
                                    [bboxes[idx[0]]], 
                                    output_size=(224, 224)).to(self.vertebra_classifier.device)
                
                vertebrae = self.vertebra_classifier(cropped)

                pred["keypoints"].append(vertebrae.keypoints.mu)
                pred["compression"].append(vertebrae.compression)
                pred["morphology"].append(vertebrae.morphology)

                pred["bbox"].append(bboxes[idx[0]])

                true["keypoints"].append(target.keypoints[target.indicator][idx[1]])
                true["compression"].append(target.compression[target.indicator][idx[1]])
                true["morphology"].append(target.morphology[target.indicator][idx[1]])

                bbox = target.boxes[target.indicator][idx[1]]
                bbox = box_ops.box_cxcywh_to_xyxy(bbox) * torch.tensor([size[1], size[0], size[1], size[0]], device=self.device)
                true["bbox"].append(bbox)

            # Using a random forest only predicting compression
            if self.random_forest_path is not None:

                predicted_keypoints = format_keypoints_for_random_forest(vertebrae.keypoints.mu)

                predicted_compressions = torch.from_numpy(self.random_forest.predict_proba(predicted_keypoints.cpu()))
                pred["compression"].append(predicted_compressions)

        # If we are using the detector to classify compression directly
        else:
            
            predicted_compressions = [p["pred_logits"][:,:-1].softmax(-1) for p in processed]

            pred["compression"].append(predicted_compressions)
            true["compression"].append([t.compression[t.indicator] for t in targets])

        # Update the true and pred dictionaries for tracking
        for key in true.keys():
            self.true[key].extend(true[key])
        
        for key in pred.keys():
            self.pred[key].extend(pred[key])

        return true, pred


    
    def on_any_test_epoch_end(self) -> None:
        """
        Compute the metrics for the test set. Used for end-to-end testing of the vertebra detection and classification.
        """

        for key in self.pred.keys():
            if key != "ids":
                self.pred[key] = torch.cat(self.pred[key], dim=0) if len(self.pred[key]) > 0 else None
                self.true[key] = torch.cat(self.true[key], dim=0) if len(self.true[key]) > 0 else None
        
        # ids = self.ids if self.level == "patient" else None
        ids = self.true["ids"]

        # Compute mean average precision
        # mean_average_precision = MeanAveragePrecision(box_format="cxcywh")
        # mAP = mean_average_precision(self.test_pred, self.test_true)

        # self.log("mAP", mAP)

        # Compute the confusion matrices
        if self.logger.log_dir is not None:
            for key in ["compression", "morphology"]:
                cm = pd.crosstab(self.true[key].cpu(), self.pred[key].argmax(-1).cpu())
                cm.to_csv(self.logger.log_dir + f"/confusion_matrix_{key}.csv")

        # For the shape classification, compute grouped metrics 
        metrics = compute_classification_metrics(
            self.true["compression"], 
            self.pred["compression"], 
            all_groups=[("normal+mild",([0, 1], [2, 3]))], 
            ids=ids)
        
        for metric in metrics:
            
            self.log_dict(metric)
            
        # Reset the test variables
        for key in self.pred.keys():
            self.pred[key] = []
            self.true[key] = []

        self.ids = []
    
    def on_test_epoch_end(self) -> None:
        return self.on_any_test_epoch_end()
    
    def on_validation_epoch_end(self) -> None:
        return self.on_any_test_epoch_end()
    
    def on_fit_start(self) -> None:
        return self.initialize_classifiers()

    def on_test_start(self) -> None:
        return self.initialize_classifiers()
    
    def on_predict_start(self) -> None:
        return self.initialize_classifiers()


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
    