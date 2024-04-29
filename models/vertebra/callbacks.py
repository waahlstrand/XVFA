
from typing import *
import torch
from torch import Tensor
import lightning as L
import matplotlib.pyplot as plt
from rich import print
import wandb
import numpy as np


class VertebraPlotCallback(L.Callback):

    def __init__(self, n_samples: int = 4, plot_frequency: int = 10):

        super().__init__()

        self.n_samples = n_samples
        self.plot_frequency = plot_frequency


    def on_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Dict[str, Tensor], batch_idx: int, name: str) -> None:
        
        if batch_idx % self.plot_frequency == 0:
            f, ax = self.plot(outputs, n_samples=self.n_samples)

            # Log image
            trainer.logger.experiment.log({
                f"{name}/plot": wandb.Image(f, caption=f"{name} plot")
            })


    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Dict[str, Tensor], batch: Any, batch_idx: int) -> None:

        self.on_batch_end(trainer, pl_module, outputs, batch_idx, "train")

    def on_validation_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Dict[str, Tensor], batch: Any, batch_idx: int) -> None:
        
        self.on_batch_end(trainer, pl_module, outputs, batch_idx, "val")

    def on_test_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Dict[str, Tensor], batch: Any, batch_idx: int, *args, **kwargs) -> None:

        if "log" in trainer.logger.experiment.__dict__:
            images = outputs["images"].detach().cpu() if outputs is not None else []

            for i, image in enumerate(images):
                f, ax = plt.subplots(1, 1, squeeze=True, dpi=300)
                self.plot_image(
                    image[0], 
                    outputs["keypoints"][i], 
                    outputs["grades"][i],
                    outputs["types"][i],
                    outputs["pred_keypoints"][i],
                    outputs["pred_keypoints_sigma"][i],
                    outputs["pred_grades"][i],
                    outputs["pred_types"][i],
                    outputs.get("loss_per_pixel", None), 
                    ax)
                trainer.logger.experiment.log({
                    f"test_stage/plot_image": wandb.Image(f, caption=f"Test plot image")
                })

                plt.close(f)

    def plot_image(self, 
                   image: Tensor, 
                   keypoints: Tensor, 
                   grades: Tensor, 
                   types: Tensor,
                   pred_keypoints: Tensor, 
                   pred_keypoints_sigma: Tensor, 
                   pred_grades: Tensor, 
                   pred_types: Tensor,
                   loss_per_pixel: Union[Tensor, None], 
                   ax: plt.Axes) -> None:


        image = image.detach().cpu().squeeze()
        keypoints_ = keypoints.detach().cpu().reshape(-1, 2)
        grades_ = grades.detach().cpu()
        types_  = types.detach().cpu()
        pred_keypoints_ = pred_keypoints.detach().cpu().reshape(-1, 2)
        pred_keypoints_sigma_ = pred_keypoints_sigma.detach().cpu().reshape(-1, 2).mean()
        pred_grades_ = pred_grades.detach().cpu().softmax(-1).argmax(-1)
        pred_types_ = pred_types.detach().cpu().softmax(-1).argmax(-1)

        # Unnormalize keypoints
        keypoints_[:, 0] *= image.shape[-1]
        keypoints_[:, 1] *= image.shape[-2]

        pred_keypoints_[:, 0] *= image.shape[-1]
        pred_keypoints_[:, 1] *= image.shape[-2]

        # Plot image
        ax.imshow(image, cmap="gray", origin="upper")

        if loss_per_pixel is not None:
            loss_per_pixel = loss_per_pixel.detach().exp().cpu()
            m = torch.max(loss_per_pixel).numpy()
            # step = 1
            # levels = np.arange(0.0, m, step) + step
            n_levels = 7
            xx, yy = np.meshgrid(np.arange(loss_per_pixel.shape[-2]), np.arange(loss_per_pixel.shape[-1]))
            for k in range(loss_per_pixel.shape[0]):
                ax.contourf(yy, xx, loss_per_pixel[k].t().cpu(), levels=n_levels, cmap="Reds", origin="upper", alpha=0.1)

        # Plot keypoints
        ax.scatter(keypoints_[:, 0], keypoints_[:, 1], s=4, color="green", label="Annotations")
        ax.scatter(pred_keypoints_[:, 0], pred_keypoints_[:,1], s=2, edgecolors="red", facecolors='none', linewidths=1.0, label="Predictions")

        # Add labels
        keypoints_ = keypoints_.reshape(-1, 6, 2)
        pred_keypoints_ = pred_keypoints_.reshape(-1, 6, 2)
            

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"g: {grades_.item()} ({pred_grades_.item()}), t: {types_.item()} ({pred_types_.item()})", fontsize=5)

    def plot(self, outputs: Dict[str, Tensor], n_samples: int): 
        """
        Plots a random sample of vertebrae from the batch on a grid, with keypoints and labels
        """

        # images, y = batch.x, batch.y
        images = outputs["images"].detach().cpu()
        n_samples = self.n_samples if self.n_samples <= images.shape[0] else images.shape[0]

        f, ax = plt.subplots(nrows=1, ncols=n_samples, squeeze=True, dpi=150)
        
        # Select a number of random samples
        idxs = torch.randperm(images.shape[0])[:n_samples]

        # Plot each sample
        for i in range(n_samples):

            idx = idxs[i].item()

            self.plot_image(
                outputs["images"][idx][0], 
                outputs["keypoints"][idx], 
                outputs["grades"][idx],
                outputs["types"][idx], 
                outputs["pred_keypoints"][idx], 
                outputs["pred_keypoints_sigma"][idx], 
                outputs["pred_grades"][idx], 
                outputs["pred_types"][idx], 
                outputs.get("loss_per_pixel", None),
                ax[i]
                )

            
        plt.subplots_adjust(wspace=0.1, hspace=0.05)
        return f, ax

                