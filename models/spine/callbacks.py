from typing import *
import torch
from torch import Tensor
import lightning as L
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from rich import print
import numpy as np
import wandb
from utils.types import Batch
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
        
        f, ax = plt.subplots(1, self.n_samples, figsize=(10, 10))

        processed = module.postprocessors["bbox"](outputs, 
                                                  torch.tensor(batch.original_sizes, 
                                                               device=module.device))
        
        idxs = np.random.choice(range(len(batch.images)), self.n_samples, replace=False)

        for i, idx in enumerate(idxs):

            image = batch.images[idx][0].cpu().numpy()

            ax[i].imshow(image, cmap="bone")

            ground_truth = batch.targets[idx].boxes[batch.targets[idx].indicator]
            ground_truth = box_ops.box_cxcywh_to_xyxy(ground_truth).cpu().numpy()
            boxes   = processed["boxes"][idx].cpu().numpy()
            labels  = processed["labels"][idx].cpu().numpy()
            size   = batch.original_sizes[idx]

            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                ax[i].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="red", lw=2))
                ax[i].text(x1, y1, f"{label}", fontsize=8, color="red")

            for box in ground_truth:
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = x1 * size[1], y1 * size[0], x2 * size[1], y2 * size[0]
                ax[i].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="green", lw=2))


            ax[i].axis("off")

        ground_truth = mpatches.Patch(color='green', label='Ground truth')
        predicted = mpatches.Patch(color='red', label='Predicted')

        plt.subplots_adjust(wspace=0.05, hspace=0.05)        
        plt.legend(handles=[ground_truth, predicted], bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=1)

        # if self.save_to_disk:
        #     f.savefig(f"plot_test.png", bbox_inches="tight")

        return f, ax