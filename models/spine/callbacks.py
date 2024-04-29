from typing import *
import torch
from torch import Tensor
import lightning as L
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from rich import print
import wandb
from data.types import Batch
from models.backbones.DINO.util import box_ops



class SpinePlotCallback(L.Callback):
    
    def __init__(self, 
                    n_samples: int = 4,
                    plot_frequency: int = 100,
                    save_to_disk: bool = False,
                    n_classes: int = 13,
                    **kwargs) -> None:
            
        super().__init__(**kwargs)
    
        self.n_samples = n_samples
        self.plot_frequency = plot_frequency
        self.save_to_disk = save_to_disk
        self.n_classes = n_classes

    def on_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Dict[str, Tensor], batch: Batch, batch_idx: int, name: str) -> None:
        
        if batch_idx % self.plot_frequency == 0:
            f, ax = self.plot(outputs, batch, pl_module)

            # Log image
            trainer.logger.experiment.log({
                f"{name}_plot": wandb.Image(f, caption=f"{name} plot")
            })

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch: Any, batch_idx: int) -> None:
        
        self.on_batch_end(trainer, pl_module, outputs, batch, batch_idx, "train")

    def on_validation_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch: Any, batch_idx: int) -> None:

        self.on_batch_end(trainer, pl_module, outputs, batch, batch_idx, "val")

    def plot(self, outputs: Dict[str, Tensor], batch: Batch, module: L.LightningModule) -> Tuple[plt.Figure, plt.Axes]:

        postprocess = module.postprocessors["bbox"]

        f, ax = plt.subplots(1, 1, squeeze=True, dpi=300)
        offset = -20

        nested, y = batch.x, batch.y

        images  = nested.tensors.detach().cpu().squeeze()
        masks   = nested.mask.detach().cpu().squeeze()
        sizes   = batch.original_sizes

        # Select a single random sample
        idx = torch.randint(0, images.shape[0], (1,)).item()

        image = images[idx][0][~masks[idx]].reshape(*sizes[idx])

        image_height, image_width = sizes[idx]
        tensor_height, tensor_width = image.shape

        # Plot image
        ax.imshow(image, cmap="gray", origin="upper")

        # Get ground truth
        bboxes      = y[idx].boxes.detach().cpu()
        indicators  = y[idx].indices.reshape(-1).detach().cpu()
        true_labels = y[idx].labels.reshape(-1).detach().cpu()

        # Remove keypoints that should not be plotted
        bboxes      = bboxes.reshape(*indicators.shape, -1)[indicators.to(bboxes.device)].reshape(-1, 4)
        true_labels = true_labels.reshape(*indicators.shape)[indicators.to(true_labels.device)].reshape(-1)

        # Unnormalize bboxes
        bboxes = box_ops.box_cxcywh_to_xyxy(bboxes)
        bboxes = bboxes * torch.tensor([image_width, image_height, image_width, image_height], dtype=bboxes.dtype)

        # Get predictions
        pred_bboxes     = outputs["pred_boxes"][idx].detach().cpu()
        processed       = postprocess(outputs, torch.tensor(sizes, device=outputs["pred_boxes"].device)) # List[Dict[str, Tensor]]
        pred_bboxes     = processed[idx]["boxes"].detach().cpu()
        pred_labels     = outputs["pred_logits"][idx].detach().cpu().softmax(-1).argmax(-1)

        # Plot bounding boxes
        for k in range(len(bboxes)):
            label = true_labels[k].item()
            box = bboxes[k]
            x1, y1, x2, y2 = box
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="green", linewidth=1))
            ax.text(x1 + offset, y1, f"{label}", color="green", fontsize=3)

        # print(pred_bboxes.shape, pred_labels.shape)
        for k in range(len(pred_bboxes)):
            label = pred_labels[k].item()
            box = pred_bboxes[k]
            x1, y1, x2, y2 = box
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=1))
            ax.text(x1 + offset, y1, f"{label}", color="red", fontsize=3)

        ground_truth = mpatches.Patch(color='green', label='Ground truth')
        predicted = mpatches.Patch(color='red', label='Predicted')

        plt.subplots_adjust(wspace=0.05, hspace=0.05)        
        plt.legend(handles=[ground_truth, predicted], bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=1)

        if self.save_to_disk:
            f.savefig(f"plot_test.png", bbox_inches="tight")

        return f, ax