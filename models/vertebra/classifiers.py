import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from data.types import *

class VertebraParameters(nn.Module):


    def __init__(self):

        super().__init__()

    def forward(self, vertebrae: Tensor) -> Dict[str, Tensor]:
        """
        A model to compute the morphological parameters of the vertebrae.

        Args:
            vertebrae (Tensor): Vertebra to compute the parameters, with anterior, middle and posterior points (B, 6, 2).

        Returns:
            Dict[str, Tensor]: Dictionary with the morphological parameters `ha`, `hp`, `hm`, `apr`, `mpr`, `mar`.
        """

        vertebrae   = vertebrae.reshape(-1, 6, 2)

        posterior   = vertebrae[:, 0:2, :]
        middle      = vertebrae[:, 2:4, :]
        anterior    = vertebrae[:, 4:6, :]

        ha = (anterior[:,0,:] - anterior[:,1,:]).norm(dim=-1)
        hp = (posterior[:,0,:] - posterior[:,1,:]).norm(dim=-1)
        hm = (middle[:,0,:] - middle[:,1,:]).norm(dim=-1)

        apr = ha / hp
        mpr = hm / hp
        mar = hm / ha

        return {
            "ha": ha,
            "hp": hp,
            "hm": hm,
            "apr": apr,
            "mpr": mpr,
            "mar": mar,
            
        }

class CrispClassifier(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        # self.tolerance = tolerance

    @torch.no_grad()
    def forward(self, vertebrae: Tensor , tolerance: float = 0.1) -> Tensor:
        """
        Classify the type of the vertebra depending on its shape.

        Args:
            vertebrae (Tensor): Vertebra to classify, with anterior, middle and posterior points (B, 6, 2).

        """
        posterior   = vertebrae[:, 0:2, :]
        middle      = vertebrae[:, 2:4, :]
        anterior    = vertebrae[:, 4:6, :]
    

        # Compute distances between points
        ha = (anterior[:,0,:] - anterior[:,1,:]).norm(dim=-1)
        hp = (posterior[:,0,:] - posterior[:,1,:]).norm(dim=-1)
        hm = (middle[:,0,:] - middle[:,1,:]).norm(dim=-1)

        apr = ha / hp # Anterior/posterior ratio (not used)
        mpr = hm / hp # Middle/posterior ratio
        mar = hm / ha # Middle/anterior ratio

        # Classify the vertebrae
        normal  = (mar <= 1 + tolerance) \
                & (mar >= 1 - tolerance) \
                & (mpr <= 1 + tolerance) \
                & (mpr >= 1 - tolerance) \
                & (apr <= 1 + tolerance) \
                & (apr >= 1 - tolerance)

        crush       = ( (mpr >= 1) & (mar <= 1) ) & (apr >= 1) & ~normal
        biconcave   = ( (mpr <= 1) & (mar <= 1) ) & ~normal & ~crush
        wedge       = ( (mpr <= 1) & (mar >= 1) ) & (apr < 1) & ~normal & ~crush & ~biconcave
        biconvex    = ( (mpr >= 1) & (mar >= 1) ) & ~normal & ~crush & ~biconcave & ~wedge

        # Set biconvex as normal
        normal = normal | biconvex

        # Create the classification tensor
        classification = torch.stack([normal, wedge, biconcave, crush], dim=-1)

        return classification

    
   

class VertebraClassifier(nn.Module):

    def __init__(self, 
                 tolerances: Union[List[float],Dict[Literal["apr", "mpr", "mar"], List[float]]] = {
                        "apr": [0.2, 0.25, 0.4],
                        "mpr": [0.2, 0.25, 0.4],
                        "mar": [0.2, 0.25, 0.4]
                 },
                 thresholds: Dict[Literal["apr", "mpr", "mar"], float] = {
                     "apr": 1.0, "mpr": 1.0, "mar": 1.0
                },
                 trainable: bool = False
                 ) -> None:
        super().__init__()

        # Make trainable
        if trainable:

            self.tolerances = nn.ParameterDict({
                k: nn.Parameter(torch.tensor(v)) for k, v in tolerances.items()
            })

            # self.tolerances = nn.Parameter(torch.tensor(tolerances))
            self.thresholds = nn.ParameterDict({
                k: nn.Parameter(torch.tensor(v)) for k, v in thresholds.items()
            })

        else:
            self.tolerances = tolerances
            self.thresholds = thresholds



    def within(self, apr: Tensor, mpr: Tensor, mar: Tensor, tolerance_idx: int = 1) -> Tensor:
        """
        Fuzzy approximation to check if the vertebrae is within the given tolerance.
        
        Args:
            apr (Tensor): Anterior/posterior ratio (N,1)
            mpr (Tensor): Middle/posterior ratio (N,1)
            mar (Tensor): Middle/anterior ratio (N,1)
            tolerance_idx (int): Index of the tolerance to use.
        """

        apr_pos_thresh = self.thresholds["apr"]*(1-self.tolerances["apr"][tolerance_idx])
        mpr_pos_thresh = self.thresholds["mpr"]*(1-self.tolerances["mpr"][tolerance_idx])
        mar_pos_thresh = self.thresholds["mar"]*(1-self.tolerances["mar"][tolerance_idx])

        apr_neg_thresh = self.thresholds["apr"]*(1+self.tolerances["apr"][tolerance_idx])
        mpr_neg_thresh = self.thresholds["mpr"]*(1+self.tolerances["mpr"][tolerance_idx])
        mar_neg_thresh = self.thresholds["mar"]*(1+self.tolerances["mar"][tolerance_idx])

        is_within, ind = torch.stack([
            self.geq(apr, apr_pos_thresh), 
            self.geq(mpr, mpr_pos_thresh), 
            self.geq(mar, mar_pos_thresh), 
            self.leq(apr, apr_neg_thresh), 
            self.leq(mpr, mpr_neg_thresh), 
            self.leq(mar, mar_neg_thresh)
            ], dim=1).min(dim=1)
        
        return is_within
    
    def geq(self, x: Tensor, value: Tensor) -> Tensor:
        """
        Fuzzy approximation to check if x is greater or equal to a value.
        
        Args:
            x (Tensor): Value to compare (N,1)
            value (Tensor): Value to compare against (N,1)
        """

        return F.sigmoid((x - value))
    
    def leq(self, x: Tensor, value: Tensor) -> Tensor:
        """
        Fuzzy approximation to check if x is less or equal to a value.

        Args:
            x (Tensor): Value to compare (N,1)
            value (Tensor): Value to compare against (N,1)
        """

        return F.sigmoid((value - x))
    
    def __call__(self, *args: Any, **kwds: Any) -> VertebraOutput:
        return super().__call__(*args, **kwds)

    def forward(self, vertebrae: Tensor) -> VertebraOutput:

        vertebrae   = vertebrae.reshape(-1, 6, 2)

        posterior   = vertebrae[:, 0:2, :]
        middle      = vertebrae[:, 2:4, :]
        anterior    = vertebrae[:, 4:6, :]

        ha = (anterior[:,0,:] - anterior[:,1,:]).norm(dim=-1)
        hp = (posterior[:,0,:] - posterior[:,1,:]).norm(dim=-1)
        hm = (middle[:,0,:] - middle[:,1,:]).norm(dim=-1)

        apr = ha / hp
        mpr = hm / hp
        mar = hm / ha

        # Can be replaced with 1 for optimal performance
        apr_pos = self.geq(apr, self.thresholds["apr"])
        mpr_pos = self.geq(mpr, self.thresholds["mpr"])
        mar_pos = self.geq(mar, self.thresholds["mar"])

        mpr_neg = self.leq(mpr, self.thresholds["apr"])
        mar_neg = self.leq(mar, self.thresholds["mpr"])
        apr_neg = self.leq(apr, self.thresholds["mar"])

        normal = self.within(apr, mpr, mar, tolerance_idx=0) # e.g. within 0.8, 1.2
        
        grad_1, ind = torch.stack([
            self.within(apr, mpr, mar, tolerance_idx=1), # e.g. within  0.75, 1.25
            1-normal
        ], dim=1).min(dim=1) # and

        grad_2, ind = torch.stack([
            self.within(apr, mpr, mar, tolerance_idx=2), # e.g.  within 0.6, 1.4
            1-normal,
            1-grad_1
        ], dim=1).min(dim=1) # and

        grad_3, ind = torch.stack([
            1-normal,
            1-grad_1,
            1-grad_2
        ], dim=1).min(dim=1) # and

        crush, ind = torch.stack([
            mpr_pos, 
            mar_neg, 
            apr_pos, 
            1-normal, 
            ], dim=1).min(dim=1) # and

        biconcave, ind = torch.stack([
            mpr_neg, 
            mar_neg, 
            1-normal, 
            1-crush, 
            ], dim=1).min(dim=1) # and

        wedge, ind = torch.stack([
            mpr_neg, 
            mar_pos, 
            apr_neg, 
            1-normal, 
            1-crush, 
            1-biconcave
            ], dim=1).min(dim=1) # and

        type_logits = torch.stack([normal, wedge, biconcave, crush], dim=-1) 

        grade_logits = torch.stack([normal, grad_1, grad_2, grad_3], dim=-1)       

        return VertebraOutput(
            grade_logits=grade_logits,
            type_logits=type_logits
        )

    
class FuzzyWedgeClassifier(VertebraClassifier):

    def __init__(self,
                 tolerances: Union[List[float],Dict[Literal["apr", "mpr", "mar"], List[float]]] = {
                        "apr": [0.2, 0.25, 0.4],
                        "mpr": [0.2, 0.25, 0.4],
                        "mar": [0.2, 0.25, 0.4]
                 },
                 thresholds: Dict[Literal["apr", "mpr", "mar"], float] = {
                     "apr": 1.0, "mpr": 1.0, "mar": 1.0
                },
                 trainable: bool = False 
                 ) -> None:
        super().__init__(tolerances=tolerances, thresholds=thresholds, trainable=trainable)

    def forward(self, vertebrae: Tensor) -> VertebraOutput:

        output = super().forward(vertebrae)

        normal      = output.type_logits[:, 0]
        wedge       = output.type_logits[:, 1]
        biconcave   = output.type_logits[:, 2]
        crush       = output.type_logits[:, 3]

        wedge_like, _  = torch.stack([wedge, crush], dim=-1).max(dim=-1)

        type_logits = torch.stack([normal, wedge_like, biconcave], dim=-1)

        return VertebraOutput(
            grade_logits=output.grade_logits,
            type_logits=type_logits
        )

