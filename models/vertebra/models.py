from typing import Optional, Union
import torch
from torch import nn
import lightning as L
from torch.optim.optimizer import Optimizer
from torchvision.models import resnet18, resnet50, swin_v2_t, Swin_V2_T_Weights, ResNet18_Weights, ResNet50_Weights
from models.vertebra.classifiers import FuzzyWedgeClassifier
from models.vertebra.criterion import RLELoss
from torch.nn.functional import mse_loss, l1_loss
from data.types import *
import torchmetrics as tm
import kornia.augmentation as K
from kornia.geometry.keypoints import Keypoints
from kornia.geometry.boxes import Boxes
from rich import print
import wandb
import matplotlib.pyplot as plt
import sklearn.metrics
from utils.evaluate import *
from utils.core import transparent_cmap

class Augmenter(nn.Module):

    def __init__(self, 
                 p_augmentation: float,
                 rotation: float = 0.1,
                 ) -> "Augmenter":
        
        super().__init__()

        self.p_augmentation = p_augmentation
        self.rotation = rotation

        self.augmenter = K.AugmentationSequential(
            K.RandomInvert(p=self.p_augmentation),
            # K.RandomHorizontalFlip(p=self.p_augmentation),
            # K.RandomVerticalFlip(p=self.p_augmentation),
            K.RandomEqualize(p=self.p_augmentation),
            K.RandomSharpness(p=self.p_augmentation),
            K.RandomMedianBlur(p=self.p_augmentation),
            K.RandomRotation(degrees=self.rotation, p=self.p_augmentation),
            data_keys=["image", "keypoints"],
        )

        self.geometric = K.AugmentationSequential(
            K.RandomAffine(degrees=self.rotation, translate=(0.1, 0.1), p=self.p_augmentation),
            K.RandomPerspective(p=self.p_augmentation),
            K.RandomElasticTransform(p=self.p_augmentation),
            K.RandomThinPlateSpline(p=self.p_augmentation), 
            data_keys=["image", "keypoints"],
        )

    def forward(self, image: Tensor, keypoints: Tensor, use_geometric: bool = False) -> Tuple[Tensor, Tensor]:

        image, keypoints = self.augmenter(image, keypoints)

        keypoints = keypoints.data

        # Normalize keypoints
        keypoints = keypoints / torch.tensor([image.shape[-1], image.shape[-2]], dtype=keypoints.dtype, device=keypoints.device)

        return image, keypoints
    
class KeypointModel(nn.Module):

    def __init__(self, dim: int, n_keypoints: int = 6, n_dim: int = 2) -> None:
        super().__init__()

        # Assume a base shape
        self.anchors = nn.Parameter(
            torch.tensor([
                [0.25, 0.2],
                [0.25, 0.8],
                [0.5, 0.2],
                [0.5, 0.8],
                [0.75, 0.2],
                [0.75,0.8],
            ]).unsqueeze(0).float(),
            requires_grad=False
        )

        self.dim = dim
        self.n_keypoints = n_keypoints
        self.n_dim = n_dim

        # Define a model to predict deviations from the base shape
        self.model = nn.Sequential(
            nn.Linear(self.dim, self.n_keypoints * self.n_dim),
            nn.Sigmoid()
        )

        self.sigma = nn.Sequential(
            nn.Linear(self.dim, self.n_keypoints * self.n_dim),
            nn.Sigmoid()
        )

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:

        # Predict the deviations from the base shape
        x      = self.model(z)
        sigma   = self.sigma(z)

        # x  = self.anchors + x.view(-1, self.n_keypoints, self.n_dim)
        # x  = torch.clamp(x, 0, 1)

        return x, sigma
    
class SingleVertebraClassifierModel(nn.Module):

    def __init__(self, 
                 n_types: int = 3, 
                 n_grades: int = 4,
                 n_keypoints: int = 6, 
                 n_dims: int = 2, 
                 model: Literal["resnet18", "resnet50", "swin_v2_t"] = "resnet18",
                 model_weights: Optional[str] = None):
        
        super().__init__()

        self.n_types = n_types
        self.n_grades = n_grades
        self.n_keypoints = n_keypoints
        self.n_dims = n_dims

        # Backbones to finetune
        match model:
            case "resnet18":
                self.features = resnet18(weights=ResNet18_Weights.DEFAULT)
                self.features.fc = nn.Linear(self.features.fc.in_features, 512)

            case "resnet50":

                self.features = resnet50(weights=ResNet50_Weights.DEFAULT)
                self.features.fc = nn.Linear(self.features.fc.in_features, 512)

            case "swin_v2_t":
                self.features = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
                self.features.head = nn.Linear(self.features.head.in_features, 512)

            case _:
                raise ValueError(f"Model {model} not supported")
            
        self.keypoint_model = KeypointModel(512, self.n_keypoints, self.n_dims)

        self.type_model = nn.Sequential(
            nn.Linear(512, self.n_types),
        )

        self.grade_model = nn.Sequential(
            nn.Linear(512, self.n_grades),
        )

        
    def forward(self, x) -> VertebraOutput:

        z               = self.features(x)
        # mu, sigma       = self.keypoint_model(z).chunk(2, dim=1)
        mu, sigma       = self.keypoint_model(z)

        type_logits     = self.type_model(z)
        grade_logits    = self.grade_model(z)

        output = VertebraOutput(
            keypoints=Prediction(mu, sigma),
            type_logits=type_logits,
            grade_logits=grade_logits,
        )

        return output

class SingleVertebraClassifier(L.LightningModule):

    def __init__(self, n_types: int = 3,
                       n_grades: int = 4, 
                       n_keypoints: int = 6, 
                    #    tolerances: List[float] = [0.2, 0.25, 0.4],
                       tolerances: Dict[Literal["apr", "mpr", "mar"], List[float]] = {"apr": [0.2, 0.25, 0.4], "mpr": [0.2, 0.25, 0.4], "mar": [0.2, 0.25, 0.4]},
                       thresholds: Dict[Literal["apr", "mpr", "mar"], float] = {"apr": 1.0, "mpr": 1.0, "mar": 1.0},
                       prior: Literal["gaussian", "laplace"] = "gaussian",
                       p_augmentation: float = 0.5,
                       rotation: float = 45.0,
                       rle_weight: float = 1.0,
                       ce_keypoint_weight: float = 1.0,
                       ce_image_weight: float = 1.0,
                       grade_weights: Optional[List[float]] = None,
                       type_weights: Optional[List[float]] = None,
                       model_name: Literal["resnet18", "swin_v2_t", "resnet50"] = "resnet18",
                       model_weights: Optional[str] = None,
                       trainable_classifier: bool = False,
                       ):
        
        super().__init__()

        self.n_types = n_types
        self.n_grades = n_grades
        self.n_keypoints = n_keypoints
        self.tolerances = tolerances
        self.thresholds = thresholds
        self.prior = prior
        self.p_augmentation = p_augmentation
        self.rotation = rotation
        self.rle_weight = rle_weight
        self.ce_keypoint_weight = ce_keypoint_weight 
        self.ce_image_weight = ce_image_weight
        self.grade_weights = torch.FloatTensor(grade_weights) if grade_weights is not None else None
        self.type_weights  = torch.FloatTensor(type_weights) if type_weights is not None else None
        self.model_name = model_name
        self.model_weights = model_weights
        self.trainable_classifier = trainable_classifier

        self.save_hyperparameters()

        self.augmentations  = nn.ModuleDict({
            "train_stage":  Augmenter(p_augmentation=p_augmentation, rotation=rotation),
            "val_stage":    Augmenter(p_augmentation=0.0),
            "test_stage":   Augmenter(p_augmentation=0.0),
        })
        
        self.model          = SingleVertebraClassifierModel(
            n_types=self.n_types,
            n_grades=self.n_grades,
            n_keypoints=self.n_keypoints, 
            model=self.model_name,
            model_weights=model_weights,
            )

        # print("Compiling model...")
        # self.model = torch.compile(self.model, mode="reduce-overhead")
        # print("Done.")

        self.classifier             = FuzzyWedgeClassifier(tolerances=self.tolerances, thresholds=self.thresholds, trainable=trainable_classifier)
        self.rle                    = RLELoss(prior=self.prior)
        self.grade_cross_entropy    = nn.CrossEntropyLoss(weight=self.grade_weights)
        self.type_cross_entropy     = nn.CrossEntropyLoss(weight=self.type_weights)
        self.distance               = mse_loss if self.prior == "gaussian" else l1_loss

        metrics = {}
        for k in ["val_stage", "test_stage"]:
            ms = {}
            for name, target in [("types", self.n_types), ("grades", self.n_grades)]:
                target_ms = {}
                for avg in ["macro", "micro", "weighted"]:
                    target_ms.update({

                            f"{name}_{avg}_accuracy": tm.Accuracy(task="multiclass", num_classes=target, average=avg),
                            f"{name}_{avg}_precision": tm.Precision(task="multiclass", num_classes=target, average=avg),
                            f"{name}_{avg}_sensitivity": tm.Recall(task="multiclass", num_classes=target, average=avg),
                            f"{name}_{avg}_specificity": tm.Specificity(task="multiclass", num_classes=target, average=avg),
                            f"{name}_{avg}_f1_score": tm.F1Score(task="multiclass", num_classes=target, average=avg),
                    })

                    if avg != "micro":
                        target_ms.update({
                            f"{name}_{avg}_auc": tm.AUROC(task="multiclass", num_classes=target, average=avg),
                            f"{name}_{avg}_average_precision": tm.AveragePrecision(task="multiclass", num_classes=target, average=avg),
                        })
                ms[name] = tm.MetricCollection(target_ms)

            metrics[k] = nn.ModuleDict(ms)

        self.metrics    = nn.ModuleDict(metrics)

        # self.test_true = {d: [] for d in range(10)}
        # self.test_pred = {d: [] for d in range(10)}
        self.test_true = []
        self.test_pred = []
        self.test_idx  = []
        self.validation_true = []
        self.validation_pred = []

    def __call__(self, *args: Any, **kwds: Any) -> VertebraOutput:
        return super().__call__(*args, **kwds)
    
    def step(self, batch: Batch, batch_idx: int, name: str = "", **kwargs) -> Dict[str, Tensor]:
        
        x, y = batch.x, batch.y
        keypoints   = torch.stack([target.keypoints for target in y]).squeeze(1)
        types       = torch.stack([target.labels for target in y]).squeeze(1)
        grades      = torch.stack([target.visual_grades for target in y]).squeeze(1)
        
        # Augment
        x, keypoints = self.augmentations[name](x, keypoints)

        # Compute keypoints and uncertainty
        output = self(x)

        # Get classifications from image features
        image_type_logits    = output.type_logits
        image_grade_logits   = output.grade_logits

        keypoints       = keypoints.reshape(*output.keypoints.mu.shape)

        # Get classification from keypoints
        keypoint_logits         = self.classifier(output.keypoints.mu)
        keypoint_type_logits    = keypoint_logits.type_logits
        keypoint_grade_logits   = keypoint_logits.grade_logits

        # Residual log-likelihood estimation, estimates the likelihood of the keypoint positions
        rle_loss            = self.rle(output.keypoints.mu, output.keypoints.sigma, keypoints)
       
        # Cross-entropy loss from keypoints
        ce_keypoint_loss    = self.type_cross_entropy(keypoint_type_logits, types)
        ce_keypoint_loss   += self.grade_cross_entropy(keypoint_grade_logits, grades)

        # Cross-entropy loss predicted from image features
        ce_image_loss       = self.type_cross_entropy(image_type_logits, types)
        ce_image_loss      += self.grade_cross_entropy(image_grade_logits, grades)

        # Total loss
        loss = self.rle_weight * rle_loss + self.ce_keypoint_weight * ce_keypoint_loss + self.ce_image_weight * ce_image_loss

        # Distance measure between mu and y
        distance = self.distance(output.keypoints.mu.detach(), keypoints.detach())

        # Get class predictions
        grades_pred = self.prediction(keypoint_grade_logits, image_grade_logits)
        types_pred  = self.prediction(keypoint_type_logits, image_type_logits)

        # Calculate mean standard deviation
        std = output.keypoints.sigma.mean()

        # Log all losses
        self.log(f"{name}/rle_loss", rle_loss, **kwargs)
        self.log(f"{name}/ce_keypoint_loss", ce_keypoint_loss, **kwargs)
        self.log(f"{name}/ce_image_loss", ce_image_loss, **kwargs)
        self.log(f"{name}/loss", loss, **kwargs)
        self.log(f"{name}/distance", distance, **kwargs)
        self.log(f"{name}/std", std, **kwargs)

        return {
            "loss": loss,
            "images": x,
            "keypoints": keypoints,
            "grades": grades,
            "types": types,
            "pred_keypoints": output.keypoints.mu,
            "pred_keypoints_sigma": output.keypoints.sigma,
            "pred_keypoint_type_logits": keypoint_type_logits,
            "pred_keypoint_grade_logits": keypoint_grade_logits,
            "pred_image_type_logits": image_type_logits,
            "pred_image_grade_logits": image_grade_logits,
            "pred_grades": grades_pred,
            "pred_types": types_pred,
        }

    def forward(self, x) -> VertebraOutput:
        return self.model(x)
    
    def naive_sample(self, images: Tensor, n_samples: int = 1000) -> Tuple[np.ndarray]:


        likelihood, xx, yy = self.get_likelihood(images)
        images      = images.cpu().numpy()
        likelihood  = likelihood.cpu().numpy()
        xx          = xx.cpu().numpy()
        yy          = yy.cpu().numpy()

        points = []
        for k in range(self.n_keypoints):
            A = np.random.uniform(0, 1, size=(n_samples, *likelihood[:,k,:,:].shape))

            # Sample the likelihood of the points
            samples = (A < likelihood[:,k,:,:]).astype(int)
            sample_idxs = np.argwhere(samples)

            # Get the x and y coordinates of the samples
            sample_x_idx, sample_y_idx = sample_idxs[:, 1], sample_idxs[:, 2]
            sample_x, sample_y = xx[sample_x_idx, sample_y_idx], yy[sample_x_idx, sample_y_idx]

            points.append(np.stack([sample_x, sample_y], axis=1))
        
        return np.stack(points, axis=1)

       

    
    def sample_multiple(self, images: Tensor, n_samples: int = 1000, chunk_size: int = 64) -> Tuple[Tensor, Tensor]:
        """
        Sample the likelihood of the keypoints from the model.
        
        Args:
            image (Tensor): The image to sample from (B, C, H, W)
            n_samples (int): The number of samples to draw
        
        Returns:
            Tuple[Tensor, Tensor]: The x and y coordinates of the samples
        """
        # points = []
        # for image in images:
        #     ps = self.sample(image, n_samples=n_samples)
        #     points.append(ps)

        
        # return torch.stack(points, dim=0)
        sampling = torch.vmap(lambda x: self.sample(x, n_samples=n_samples), in_dims=0, out_dims=0, chunk_size=chunk_size, randomness="different")

        points = sampling(images)

        return points
    
    def sample(self, image: Tensor, n_samples: int = 1000) -> Tuple[Tensor, Tensor]:
        """
        Sample the likelihood of the keypoints from the model.
        
        Args:
            image (Tensor): The image to sample from (1, C, H, W)
            n_samples (int): The number of samples to draw
            
        Returns:
            Tuple[Tensor, Tensor]: The x and y coordinates of the samples
        """
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        likelihood, xx, yy = self.get_likelihood(image)
        # likelihood = likelihood.cpu().numpy()
        # xx = xx.cpu().numpy()
        # yy = yy.cpu().numpy()

        points = []
        likelihood = likelihood.squeeze()

        # Loop over keypoints
        for i in range(self.n_keypoints):
            l = likelihood[i, :, :]
            # sample = np.random.choice(
                # a = np.arange(0, len(l.flatten())), 
                # size = n_samples, 
                # p = l.flatten(), 
                # replace=True
                # )
            
            # sample_x_idx, sample_y_idx = np.unravel_index(sample, l.shape)
            # sample_x, sample_y = xx[sample_x_idx, sample_y_idx], yy[sample_x_idx, sample_y_idx]

            samples = torch.multinomial(l.flatten(), n_samples, replacement=True)
            sample_x_idx, sample_y_idx = torch.unravel_index(samples, l.shape)
            sample_x, sample_y = xx[sample_x_idx, sample_y_idx], yy[sample_x_idx, sample_y_idx]

            points.append(torch.stack([sample_x, sample_y], dim=1))

        return torch.stack(points, dim=1)
    
    def get_likelihood(self, image: Tensor, n_points: int = 224) -> Tuple[Tensor, Tensor, Tensor]:

        if n_points is None:
            n_points = image.shape[-1]

        # Create a grid of points over the image
        x = torch.linspace(0, 1, n_points, device=image.device)
        y = torch.linspace(0, 1, n_points, device=image.device)
        xx, yy = torch.meshgrid(x, y)
        points = torch.stack([yy.flatten(), xx.flatten()], dim=1).to(image.device) # (H * W, 2)

        output = self(image) # VertebraOutputs (mu, sigma) for each keypoint (B, K, 2), (B, K, 2)
        loss = self.rle.inference(output.keypoints.mu, output.keypoints.sigma, points) # (B x H x W, K)
        likelihood = (-loss).exp() / (-loss).exp().sum(dim=(-1,-2),keepdims=True)

        return likelihood, xx, yy


    def visualize_uncertainty(self, image: Tensor) -> Tensor:

        likelihood, xx, yy = self.get_likelihood(image)
        mycmap = transparent_cmap(plt.cm.Reds)

        # Plot the likelihood of the points
        f, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.imshow(image[0].squeeze().cpu().numpy(), cmap="gray")

        for keypoint in range(self.n_keypoints):
            ax.contourf(
                xx.numpy()*image.shape[-1],
                yy.numpy()*image.shape[-2],
                likelihood[keypoint, :, :].squeeze().cpu().numpy(),
                15,
                cmap=mycmap,
                vmin=0,
                vmax=likelihood[keypoint, :, :].max().item()
            )

    
    def training_step(self, batch: Batch, batch_idx: int) -> Dict[str, Tensor]:
        output = self.step(batch, batch_idx, name="train_stage", prog_bar=False, on_epoch=True, on_step=True, batch_size=batch.x.shape[0])

        return output
    
    def validation_step(self, batch: Batch, batch_idx: int) -> Dict[str, Tensor]:
        
        output = self.step(batch, batch_idx, name="val_stage", prog_bar=False, on_epoch=True, on_step=False, batch_size=batch.x.shape[0])

        val_types_true = output["types"]
        val_grades_true = output["grades"]

        val_types_pred = output["pred_types"]
        val_grades_pred = output["pred_grades"]

        self.validation_true.append((val_types_true, val_grades_true))
        self.validation_pred.append((val_types_pred, val_grades_pred))

        return output
    
    def test_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, Tensor]:

        output = self.step(batch, batch_idx, name="test_stage", prog_bar=False, on_epoch=True, on_step=False, batch_size=batch.x.shape[0])

        test_types_true = output["types"]
        test_grades_true = output["grades"]

        test_types_pred = output["pred_types"]
        test_grades_pred = output["pred_grades"]

        self.test_true.append((test_types_true, test_grades_true))
        self.test_pred.append((test_types_pred, test_grades_pred))

        return output

    def on_validation_epoch_end(self) -> None:

        # Log classifier thresholds
        for k, v in self.classifier.tolerances.items():
            for name, val in (("mild", 0), ("moderate", 1), ("severe", 2)):
                self.log(f"tolerance/{k}/{name}", v[val].detach().item(), prog_bar=False, on_epoch=True, on_step=False)

        # Log classifier tolerances
        for k, v in self.classifier.thresholds.items():
            self.log(f"threshold/{k}", v.detach().item(), prog_bar=False, on_epoch=True, on_step=False)

        type_labels = ["normal", "wedge", "biconcave"]
        grade_labels = ["normal", "grade 1", "grade 2", "grade 3"]

        val_types_true, val_grades_true = zip(*self.validation_true)
        val_types_true = torch.cat(val_types_true, dim=0).to(self.device)
        val_grades_true = torch.cat(val_grades_true, dim=0).to(self.device)

        val_types_pred, val_grades_pred = zip(*self.validation_pred)
        val_types_pred = torch.cat(val_types_pred, dim=0).to(self.device)
        val_grades_pred = torch.cat(val_grades_pred, dim=0).to(self.device)
            
        try:
            self.on_any_test_end(val_types_true, val_types_pred, name="val_stage", target="types", labels=type_labels)
            self.on_any_test_end(val_grades_true, val_grades_pred, name="val_stage", target="grades", labels=grade_labels)
        except Exception as e:
            pass

        self.validation_true = []
        self.validation_pred = []

    def on_test_epoch_end(self) -> None:

        type_labels = ["normal", "wedge", "biconcave"]
        grade_labels = ["normal", "grade 1", "grade 2", "grade 3"]

    
        test_types_true, test_grades_true = zip(*self.test_true)
        test_types_true = torch.cat(test_types_true, dim=0).to(self.device)
        test_grades_true = torch.cat(test_grades_true, dim=0).to(self.device)

        test_types_pred, test_grades_pred = zip(*self.test_pred)
        test_types_pred = torch.cat(test_types_pred, dim=0).to(self.device)
        test_grades_pred = torch.cat(test_grades_pred, dim=0).to(self.device)
        
        try:
            self.on_any_test_end(test_types_true, test_types_pred, name=f"test_stage", target="types", labels=type_labels)
            self.on_any_test_end(test_grades_true, test_grades_pred, name=f"test_stage", target="grades", labels=grade_labels)
        except Exception as e:
            pass

        self.test_true = []
        self.test_pred = []

    def prediction(self, keypoint_logits: Optional[Tensor], image_logits: Optional[Tensor]):

        if not (self.ce_image_weight > 0):
            pred = keypoint_logits.softmax(dim=1)

        # If we have no keypoint classification, we use the image classification
        elif not (self.ce_keypoint_weight > 0):
            pred = image_logits.softmax(dim=1)

        else: 
            pred = image_logits.softmax(dim=1) * keypoint_logits.softmax(dim=1)

        return pred

    def on_any_test_end(self, 
                        trues: Tensor, 
                        preds: Tensor, 
                        name: str, 
                        target: str, 
                        labels=[]) -> Dict[str, Tensor]:
        
        trues = trues.squeeze().cpu().numpy()
        preds = preds.cpu().numpy()
        
        if preds.shape[-1] == 4:
            all_groups =[
                ("normal", ([0], [1, 2, 3])),
                ("mild",([1], [0, 2, 3])),
                ("moderate",([2], [0, 1, 3])),
                ("severe",([3], [0, 1, 2])),
                ("normal+mild",([0, 1], [2, 3])),
            ]
        elif preds.shape[-1] == 3:
            all_groups = [
                ("normal", ([0, ], [1, 2])),
                ("wedge", ([1, ], [0, 2])),
                ("concave", ([2, ], [0, 1])),
            ]
        else:
            raise ValueError(f"Number of classes {preds.shape[-1]} not supported")
        
        for group_name, groups in all_groups:
            # Compute ROC curve for a multi-class classification problem using the One-vs-Rest (OvR) strategy
            trues_binary, preds_grouped = grouped_classes(trues, preds, groups, n_classes=preds.shape[-1])
            # print(trues_binary.sum(), group_name)

            roc = grouped_roc_ovr(trues, preds, groups, n_classes=preds.shape[-1])
            
            # Compute relevant metrics
            auc     = roc["roc_auc"]
            youden  = roc["youden_threshold"]
            preds_thresh   = (preds_grouped > youden).astype(int)

            # Compute confusion matrix
            cm = sklearn.metrics.confusion_matrix(trues_binary, preds_thresh, labels=[0,1])

            # Compute metrics
            # Sensitivity, specificity, precision, f1-score
            sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
            precision   = cm[1, 1] / (cm[1, 1] + cm[0, 1])
            accuracy    = (cm[0, 0] + cm[1, 1]) / cm.sum()

            # Get the prevalence of the positive class
            prevalence = len(trues_binary) - trues_binary.sum()  

            f1_score    = 2 * (precision * sensitivity) / (precision + sensitivity)

            # Log metrics
            self.log(f"{name}/{target}/{group_name}/auc", auc, prog_bar=False, on_epoch=True, on_step=False)
            self.log(f"{name}/{target}/{group_name}/youden", youden, prog_bar=False, on_epoch=True, on_step=False)
            self.log(f"{name}/{target}/{group_name}/sensitivity", sensitivity, prog_bar=False, on_epoch=True, on_step=False)
            self.log(f"{name}/{target}/{group_name}/specificity", specificity, prog_bar=False, on_epoch=True, on_step=False)
            self.log(f"{name}/{target}/{group_name}/precision", precision, prog_bar=False, on_epoch=True, on_step=False)
            self.log(f"{name}/{target}/{group_name}/accuracy", accuracy, prog_bar=False, on_epoch=True, on_step=False)
            self.log(f"{name}/{target}/{group_name}/f1_score", f1_score, prog_bar=False, on_epoch=True, on_step=False)
            self.log(f"{name}/{target}/{group_name}/prevalence", prevalence, prog_bar=False, on_epoch=True, on_step=False)

        return {
                "auc": torch.tensor(auc),
                "youden": torch.tensor(youden),
                "sensitivity": torch.tensor(sensitivity),
                "specificity": torch.tensor(specificity),
                "precision": torch.tensor(precision),
                "accuracy": torch.tensor(accuracy),
                "f1_score": torch.tensor(f1_score),
            }

    