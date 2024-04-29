from models.backbones.realnvp.models import RealNVP
from torch import Tensor
import torch
import torch.nn as nn
import math

from typing import *


class RLELoss(nn.Module):

    def __init__(self, n_keypoints: int = 6, n_dims: int = 2, prior: Literal["laplace", "gaussian"] = "laplace") -> None:
        
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