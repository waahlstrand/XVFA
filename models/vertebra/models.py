from typing import Optional
import torch
from torch import nn
import lightning as L
from torchvision.models import resnet18, resnet50, swin_v2_t, Swin_V2_T_Weights, ResNet18_Weights, ResNet50_Weights
from models.vertebra.classifiers import FuzzyWedgeClassifier
from models.vertebra.criterion import VertebraLoss
from torch.nn.functional import mse_loss, l1_loss
from utils.types import *
import torchmetrics as tm
import kornia.augmentation as K
import matplotlib.pyplot as plt
import sklearn.metrics
from utils.evaluate import *


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

    def __call__(self, *args: Any, **kwds: Any) -> Tuple[Tensor, Tensor]:
        return super().__call__(*args, **kwds)       

    def forward(self, image: Tensor, keypoints: Tensor, use_geometric: bool = False) -> Tuple[Tensor, Tensor]:

        image, keypoints = self.augmenter(image, keypoints)

        keypoints = keypoints.data

        # Normalize keypoints
        keypoints = keypoints / torch.tensor([image.shape[-1], image.shape[-2]], dtype=keypoints.dtype, device=keypoints.device)

        return image, keypoints
    
class KeypointModel(nn.Module):

    def __init__(self, dim: int, n_keypoints: int = 6, n_dim: int = 2) -> None:
        super().__init__()

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

    def __call__(self, *args: Any, **kwds: Any) -> Tuple[Tensor, Tensor]:
        return super().__call__(*args, **kwds)

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:

        # Predict the mode and scale of the keypoints
        x      = self.model(z)
        sigma   = self.sigma(z)

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

    def __call__(self, *args: Any, **kwds: Any) -> Tuple[PointPrediction, Tensor, Tensor]:
        return super().__call__(*args, **kwds)

        
    def forward(self, x) -> Tuple[PointPrediction, Tensor, Tensor]:

        z               = self.features(x)
        
        mu, sigma       = self.keypoint_model(z)

        type_logits     = self.type_model(z)
        grade_logits    = self.grade_model(z)


        return PointPrediction(mu=mu, sigma=sigma), grade_logits, type_logits

class SingleVertebraClassifier(L.LightningModule):

    def __init__(self, n_types: int = 3,
                       n_grades: int = 4, 
                       n_keypoints: int = 6, 
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
        
        # Image-based model
        self.model          = SingleVertebraClassifierModel(
            n_types=self.n_types,
            n_grades=self.n_grades,
            n_keypoints=self.n_keypoints, 
            model=self.model_name,
            model_weights=model_weights,
            )

        # Keypoint-based classifier
        self.classifier     = FuzzyWedgeClassifier(tolerances=self.tolerances, thresholds=self.thresholds, trainable=trainable_classifier)

        # Loss function
        self.vertebra_loss  = VertebraLoss(n_keypoints=self.n_keypoints, 
                                           n_dims=2, prior=self.prior, 
                                           compression_weights=self.grade_weights, 
                                           morphology_weights=self.type_weights, 
                                           rle_weight=self.rle_weight, 
                                           ce_image_weight=self.ce_image_weight, 
                                           ce_keypoint_weight=self.ce_keypoint_weight)
        
        # Compatibility
        # self.rle = self.vertebra_loss.rle
        # self.grade_cross_entropy = self.vertebra_loss.ce_grade
        # self.type_cross_entropy = self.vertebra_loss.ce_type

        # Define metrics used for classification
        self.distance       = mse_loss if self.prior == "gaussian" else l1_loss
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

        self.test_true = []
        self.test_pred = []
        self.test_idx  = []
        self.validation_true = []
        self.validation_pred = []

    def __call__(self, *args: Any, **kwds: Any) -> VertebraPrediction:
        return super().__call__(*args, **kwds)
    
    def step(self, batch: Batch, batch_idx: int, name: str = "", **kwargs) -> VertebraModelOutput:
        
        images = batch.images
        keypoints, compressions, morphologies = batch.keypoints, batch.compressions, batch.morphologies
        
        # Augment the images with keypoints
        x, keypoints = self.augmentations[name](images, keypoints)

        # Compute keypoints and uncertainty
        prediction = self(images)

        keypoints       = keypoints.reshape(*prediction.keypoints.mu.shape)

        # Compute the loss
        loss = self.vertebra_loss(prediction, keypoints, compressions, morphologies)

        # Distance measure between mu and y
        distance = self.distance(prediction.keypoints.mu.detach(), keypoints.detach())

        # Calculate mean standard deviation
        std = prediction.keypoints.sigma.mean()

        # Log all losses
        self.log(f"{name}/loss", loss, **kwargs)
        self.log(f"{name}/distance", distance, **kwargs)
        self.log(f"{name}/std", std, **kwargs)

        return VertebraModelOutput(
            loss=loss,
            images=images,
            true=VertebraOutput(
                keypoints=PointPrediction(mu=keypoints), 
                compression=compressions, 
                morphology=morphologies
                ),
            prediction=VertebraOutput(
                keypoints=prediction.keypoints, 
                compression=prediction.compression, 
                morphology=prediction.morphology
                )

        )

    def forward(self, x) -> VertebraPrediction:
        """
        Main forward pass of the model. Predicts the keypoints, grades and types of the vertebrae,
        based on the image and the keypoints.

        Args:
            x (Tensor): The input image (B, C, H, W)

        Returns:
            VertebraPrediction: The prediction of the model
        """

        # Predict p(x | I) and p(c | I)
        keypoints, img_compression_logits, img_morphology_logits = self.model(x)

        # Predict p(c | x)
        kp_compression_logits, kp_morphology_logits = self.classifier(keypoints.mu)

        ## For grades:
        compression = self.combined_classification(kp_compression_logits, img_compression_logits)

        ## For types:
        morphology  = self.combined_classification(kp_morphology_logits, img_morphology_logits)

        return VertebraPrediction(
            keypoints=PointPrediction(mu=keypoints.mu, sigma=keypoints.sigma),
            compression=compression,
            morphology=morphology,
            image_logits=ClassPrediction(compression=img_compression_logits, morphology=img_morphology_logits),
            keypoint_logits=ClassPrediction(compression=kp_compression_logits, morphology=kp_morphology_logits)
        )  


    def combined_classification(self, keypoint_logits: Optional[Tensor], image_logits: Optional[Tensor]):

        if not (self.ce_image_weight > 0):
            pred = keypoint_logits.softmax(dim=1)

        # If we have no keypoint classification, we use the image classification
        elif not (self.ce_keypoint_weight > 0):
            pred = image_logits.softmax(dim=1)

        else: 
            pred = image_logits.softmax(dim=1) * keypoint_logits.softmax(dim=1)

        return pred
        
    def training_step(self, batch: Batch, batch_idx: int) -> Dict[str, Tensor]:
        output = self.step(batch, batch_idx, name="train_stage", prog_bar=False, on_epoch=True, on_step=True, batch_size=batch.images.shape[0])

        return output
    
    def validation_step(self, batch: Batch, batch_idx: int) -> Dict[str, Tensor]:
        
        output = self.step(batch, batch_idx, name="val_stage", prog_bar=False, on_epoch=True, on_step=False, batch_size=batch.images.shape[0])

        self.validation_true.append((output.true.morphology, output.true.compression))
        self.validation_pred.append((output.prediction.morphology, output.prediction.compression))

        return output
    
    def test_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, Tensor]:

        output = self.step(batch, batch_idx, name="test_stage", prog_bar=False, on_epoch=True, on_step=False, batch_size=batch.images.shape[0])

        self.test_true.append((output.true.morphology, output.true.compression))
        self.test_pred.append((output.prediction.morphology, output.prediction.compression))

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

    

class LikelihoodVisualizer(nn.Module):

    def __init__(self, model: SingleVertebraClassifier, n_keypoints: int = 6) -> None:
        super().__init__()
        self.model = model
        self.n_keypoints = n_keypoints

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

            print(sample_x.shape, sample_y.shape)

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

        points = []
        likelihood = likelihood.squeeze()

        # Loop over keypoints
        for i in range(self.n_keypoints):
            l = likelihood[i, :, :]

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

        output = self.model(image) 
        loss = self.model.vertebra_loss.rle.inference(output.keypoints.mu, output.keypoints.sigma, points) # (B x H x W, K)
        likelihood = (-loss).exp() / (-loss).exp().sum(dim=(-1,-2),keepdims=True)

        return likelihood, xx, yy


    def visualize_uncertainty(self, images: Tensor, **kwargs) -> List[Tuple[plt.Figure, plt.Axes]]:

        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        likelihood, xx, yy = self.get_likelihood(images)
        mycmap = transparent_cmap(plt.cm.Reds)

        # Plot the likelihood of the points
        fs, axs = [], []

        for image, ll in zip(images, likelihood):
            f, ax = plt.subplots(1, 1, **kwargs)

            ax.imshow(image[0].squeeze().cpu().numpy(), cmap="gray")

            for keypoint in range(self.n_keypoints):
                ax.contourf(
                    yy.detach().cpu().numpy()*image.shape[-2],
                    xx.detach().cpu().numpy()*image.shape[-1],
                    ll[keypoint, :, :].squeeze().cpu().numpy(),
                    15,
                    cmap=mycmap,
                    vmin=0,
                    vmax=ll[keypoint, :, :].max().item()
                )

            fs.append(f)
            axs.append(ax)

        return fs, axs