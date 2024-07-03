from models.backbones.realnvp.models import RealNVP
from torch import Tensor
import torch
import torch.nn as nn
import math
from utils.types import *
from typing import *

class VertebraLoss(nn.Module):

    def __init__(self, 
                 n_keypoints: int = 6, 
                 n_dims: int = 2, 
                 prior: Literal["laplace", "gaussian"] = "laplace",
                 compression_weights: Optional[List[float]] = None,
                 morphology_weights: Optional[List[float]] = None,
                 rle_weight: float = 1.0, 
                 ce_image_weight: float = 1.0,
                 ce_keypoint_weight: float = 1.0,
                 ) -> None:
        """
        The loss function for the fine vertebra model. The loss function is a weighted sum of the following components:
        
        1. The RLE loss: The loss function for the keypoint prediction. The RLE loss is the negative log-likelihood of the keypoint
        where the likelihood is calculated using a normalizing flow. 
        2. The cross-entropy loss for the classification based on image features
        3. The cross-entropy loss for the classification based on keypoints

        Args:
            n_keypoints (int, optional): The number of keypoints to predict. Defaults to 6.
            n_dims (int, optional): The number of dimensions of the keypoints. Defaults to 2.
            prior (Literal["laplace", "gaussian"], optional): The prior distribution for the RLE loss. Defaults to "laplace".
            compression_weights (Optional[List[float]], optional): The weights for the grade classification. Defaults to None.
            morphology_weights (Optional[List[float]], optional): The weights for the morphology classification. Defaults to None.
            rle_weight (float, optional): The weight for the RLE loss. Defaults to 1.0.
            ce_image_weight (float, optional): The weight for the cross-entropy loss for the image features. Defaults to 1.0.
            ce_keypoint_weight (float, optional): The weight for the cross-entropy loss for the keypoints. Defaults to 1.0.
        """
        
        super().__init__()

        self.rle_weight = rle_weight
        self.ce_image_weight = ce_image_weight
        self.ce_keypoint_weight = ce_keypoint_weight
        self.compression_weights = torch.FloatTensor(compression_weights) if compression_weights is not None else None
        self.morphology_weights  = torch.FloatTensor(morphology_weights) if morphology_weights is not None else None
        self.rle = RLELoss(n_keypoints=n_keypoints, n_dims=n_dims, prior=prior)
        self.ce_compression = nn.CrossEntropyLoss(weight=self.compression_weights)
        self.ce_morphology = nn.CrossEntropyLoss(weight=self.morphology_weights)


    def forward(self,
                prediction: VertebraPrediction,
                keypoints: Tensor,
                compressions: Tensor,
                morphologies: Tensor,
                ) -> Tensor:
        """
        Calculate the loss for the fine vertebra model.

        Args:
            prediction (VertebraPrediction): The prediction from the model
            keypoints (Tensor): The ground truth keypoints
            compressions (Tensor): The ground truth compressions
            morphologies (Tensor): The ground truth morphologies
        
        Returns:
            Tensor: The loss
        """
        
        rle = self.rle(prediction.keypoints.mu, prediction.keypoints.sigma, keypoints)

        ce_keypoint_loss    = self.ce_morphology(prediction.keypoint_logits.compression, morphologies)
        ce_keypoint_loss   += self.ce_compression(prediction.keypoint_logits.morphology, compressions)

        ce_image_loss       = self.ce_morphology(prediction.image_logits.compression, morphologies)
        ce_image_loss      += self.ce_compression(prediction.image_logits.morphology, compressions)

        loss =  self.rle_weight * rle + \
                self.ce_image_weight * ce_image_loss + \
                self.ce_keypoint_weight * ce_keypoint_loss

        return loss

class RLELoss(nn.Module):

    def __init__(self, n_keypoints: int = 6, n_dims: int = 2, prior: Literal["laplace", "gaussian"] = "laplace") -> None:
        """
        The loss function for the RLE prediction. The loss function is the negative log-likelihood of the keypoint
        where the likelihood is calculated using a normalizing flow.

        Args:
            n_keypoints (int, optional): The number of keypoints to predict. Defaults to 6.
            n_dims (int, optional): The number of dimensions of the keypoints. Defaults to 2.
            prior (Literal["laplace", "gaussian"], optional): The prior distribution for the RLE loss. Defaults to "laplace".
        """
        
        super().__init__()
        self.eps = 1e-9
        self.flow = RealNVP()
        self.prior = prior
        self.n_keypoints = n_keypoints
        self.n_dims = n_dims

        self.log_phi = torch.vmap(self.flow.log_prob, in_dims=0, out_dims=0, chunk_size=512)

    def forward(self, mu: Tensor, sigma: Tensor, x: Tensor) -> Tensor:
        """
        Args:
            mu (Tensor): (B, K x 2) The predicted mean of the distribution
            sigma (Tensor): (B, K x 2) The predicted standard deviation of the distribution
            x (Tensor): (N, 2) The query point, or the ground truth point in the case of training

        Returns:
            Tensor: The log-likelihood of the query point under the distribution
        """
        
        mu, sigma, x = mu.reshape(-1, self.n_keypoints, self.n_dims), sigma.reshape(-1, self.n_keypoints, self.n_dims), x.reshape(-1, self.n_keypoints, self.n_dims)

        # Calculate the deviation from a sample x
        error = (mu - x) / (sigma + self.eps) # (B x K, N, D)
        log_phi = self.flow.log_prob(error.view(-1, self.n_dims)).view(-1, self.n_keypoints, 1)
        log_sigma = torch.log(sigma).view(-1, self.n_keypoints, self.n_dims)

        match self.prior:
            case "laplace":
                log_q = torch.log(2 * sigma) + torch.abs(error)

            case "gaussian":
                log_q = torch.log(sigma * math.sqrt(2 * math.pi)) + 0.5 * error**2

            case _:
                raise NotImplementedError("Prior not implemented")
            
        nll = log_sigma - log_phi + log_q
        nll /= len(nll)
        nll = nll.sum()

        return nll
    
    @torch.no_grad()
    def inference(self, mu: Tensor, sigma: Tensor, x: Tensor) -> Tensor:
        """
        Calculate the log-likelihood of a query point under the distribution
        
        Args:
            mu (Tensor): (B, K x 2) The predicted mean of the distribution
            sigma (Tensor): (B, K x 2) The predicted standard deviation of the distribution
            x (Tensor): (N x N, 2) The query point, or the ground truth point in the case of training

        Returns:
            Tensor: The log-likelihood of the query point under the distribution (B, K, N, N)
        """

        # Calculate the log-likelihood from the flow
        n_points    = int(math.sqrt(x.shape[0]))
        mu          = mu.reshape(-1, self.n_keypoints, 1, self.n_dims)
        sigma       = sigma.reshape(-1, self.n_keypoints, 1, self.n_dims)
        x           = x.reshape(-1, self.n_dims)

        error = (mu - x) / (sigma + self.eps) 

        # Compute the log probability of the error under the flow
        log_phi = self.log_phi(error.view(-1, self.n_keypoints, self.n_dims)).view(-1, self.n_keypoints, n_points)
        
        log_sigma = torch.log(sigma)

        match self.prior:
            case "laplace":
                log_q = torch.log(2 * sigma) + torch.abs(error) # (B x K, N, D)
            case "gaussian":
                log_q = torch.log(sigma * math.sqrt(2 * math.pi)) + 0.5 * error**2
            case _:
                raise NotImplementedError("Prior not implemented")
            

        log_phi = log_phi.view(-1, self.n_keypoints, n_points ** 2)
        log_sigma = log_sigma.sum(-1).repeat(1, 1, n_points ** 2)
        log_q = log_q.sum(-1).view(-1, self.n_keypoints, n_points ** 2)

        nll = log_sigma - log_phi + log_q 
        nll = nll.reshape(-1, self.n_keypoints, n_points, n_points)

        return nll