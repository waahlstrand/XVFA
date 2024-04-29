from typing import *
from data.types import Batch, Output, Loss, Prediction
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning as L
import torchvision.models as models
from models.backbones.detr.models.detr import build
from models.criterion import RLELoss
from dataclasses import asdict
from models.backbones.detr.util.misc import NestedTensor
from models.criterion import PolynomialPenalty, SetCriterion
from models.backbones.detr.models.detr import build_backbone, build_transformer, build_matcher
from models.backbones.detr.models.detr import DETR, PostProcess
from models.base import Detector, MLP




class VertebraDetr(Detector):

    def __init__(self, 
                 args, 
                 n_classes: int = 4, 
                 n_keypoints: int = 6, 
                 n_dim: int = 2, 
                 n_channels: int = 3,
                 class_weights: List[float] = None,
                 **kwargs) -> None:

        self.n_keypoints = n_keypoints
        self.n_dim = n_dim
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.class_weights = class_weights

        num_queries = args.num_queries
        
        args.num_queries = 100
        detector, criterion, postprocessors = build_detr(args, class_weights=class_weights)

        super().__init__(args.lr, args.lr_backbone, args.weight_decay, **kwargs)
        
        self.model = detector
        self.model.class_embed = nn.Linear(self.model.class_embed.in_features, 91 + 1)

        if args.frozen_weights:
            checkpoint = torch.load(args.frozen_weights, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"], strict=False)
            # self.model.eval()
            # for param in self.model.parameters():
            #     param.requires_grad = False
        # print(num_queries, args.num_queries)
        if num_queries != 100:
            self.model.query_embed = nn.Embedding(num_queries, self.model.transformer.d_model)

        # Only predict the objectness logits
        self.model.class_embed = nn.Linear(self.model.class_embed.in_features, 1)
        self.model.num_queries = num_queries

        try:
            if not args.debug:
                self.model = torch.compile(self.model, mode="reduce-overhead")
        except:
            raise ValueError("The model is not compilable.")


        self.criterion = criterion
        self.postprocessors = postprocessors
        self.poly_weight = args.polynomial_loss_coef


    def forward(self, x: NestedTensor) -> Dict[str, Tensor]:

        x = self.model(x) # Dict[str, Tensor]

        return x

    def step(self, batch: Batch, batch_idx: int, name: str = "", **kwargs) -> Output:

        x, y = batch.x, batch.y

        output = self(x)

        # Convert the targets to the format expected by the criterion
        y = [{k: v for k, v in asdict(_).items() if k in ["boxes", "keypoints", "labels", "indices"]} for _ in y]

        loss_dict = self.criterion(output, y)

        # Logging
        losses = [
            "loss_ce",
            "loss_bbox",
            "loss_giou",
            "loss_keypoints",
            "loss_polynomial",
            "class_error",
            "cardinality_error",
            "mae_keypoints",
            "auroc",
            "f1_score",
            "accuracy",
        ]
        for loss_name in losses:

            self.log(f"{name}/{loss_name}", loss_dict[loss_name], **kwargs)

        output = Output(
            bboxes=Prediction(mu=output["pred_boxes"], sigma=output["pred_sigma_boxes"]),
            keypoints=Prediction(mu=output["pred_keypoints"], sigma=output["pred_sigma_keypoints"]),
            logits=output["pred_logits"],
        )

        return loss_dict, output

    def build_metrics(self) -> nn.ModuleDict:

        regression = torchmetrics.MetricCollection({
            'mse': torchmetrics.MeanSquaredError(),
            'mape': torchmetrics.MeanAbsolutePercentageError(),
        })

        classification = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=self.n_classes, average="weighted"),
            'f1': torchmetrics.F1Score(task="multiclass", num_classes=self.n_classes, average="weighted"),
            'auroc': torchmetrics.AUROC(task="multiclass", num_classes=self.n_classes, average="weighted"),
        })

        metrics = nn.ModuleDict({
            'regression': regression,
            'classification': classification,
        })


        return metrics


def build_detr(args, class_weights: List[float] = None):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250

    elif args.dataset_file == "superb":

        num_classes = args.n_classes

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )

    matcher = build_matcher(args)
    weight_dict = {
        'loss_ce': 1, 
        'loss_bbox': args.bbox_loss_coef,
        'loss_keypoints': args.keypoint_loss_coef,
        'loss_giou': args.giou_loss_coef,
        'loss_polynomial': args.polynomial_loss_coef,
        }

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'keypoints']

    criterion = SetCriterion(num_classes,
                            matcher=matcher, 
                            weight_dict=weight_dict,
                            eos_coef=args.eos_coef, 
                            losses=losses, 
                            class_weights=class_weights,
                            batch_size=args.batch_size,
                            n_vertebrae=args.n_vertebrae,
                            n_keypoints=args.n_keypoints,
                            n_dims=args.n_dims,
                            missing_weight=args.missing_weight,
                            )
    # criterion.to(model.device)
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors



# class FlowLikelihood(nn.Module):

#     def __init__(self, model: VertebraDetr) -> None:
#         super().__init__()

#         self.model = model

#     def forward(self, x: Tensor) -> Tensor:

#         batch_size, n_channels, H, W = x.shape

#         output = self.model(x)

#         mu, sigma = output["pred_keypoints"], output["pred_sigma_keypoints"]

#         mu, sigma = mu.view(batch_size, -1, 2), sigma.view(batch_size, -1, 2)

#         Y, X = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W))
#         x = torch.stack([X, Y], dim=-1).reshape(-1, 2).to(self.model.device)

#         eps = 1e-9
#         for idx in range(batch_size):
#             error = (mu[idx, :, None] - x) / (sigma[idx, :, None] + eps)

#             log_phi = self.model.criterion.flow.log_prob(error.view(-1, 2)).view(-1, 20, 6, 1)
#             log_q   = torch.log(2*sigma) + torch.abs(error)
#             log_likelihood = (torch.log(sigma) - log_phi + log_q).sum(dim=-1)
            