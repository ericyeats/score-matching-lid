from .base import LIDEstimator
from ..methods.denoiser import Denoiser
import torch
from copy import deepcopy

EIG_EPS = 1e-2

class ParametricLIDEstimator(LIDEstimator):

    def __init__(self, model: Denoiser, **lid_kwargs):
        super().__init__()
        self.model = model
        self.lid_kwargs = deepcopy(lid_kwargs)

class DenoisingLossLIDEstimator(ParametricLIDEstimator):

    def estimate_lid(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.lid_kwargs.pop('estimate_tangent_subspace', None)
        lid_estimates = self.model.estimate_lid_denoising_loss(x, **self.lid_kwargs)
        return lid_estimates
    
class EigenvalueLIDEstimator(ParametricLIDEstimator):

    def estimate_lid(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.lid_kwargs.pop('estimate_tangent_subspace', None)
        _, eigvals, _ = self.model.estimate_lid_subspace(x, estimate_tangent_subspace=True, **self.lid_kwargs)
        lid_estimates = (eigvals > EIG_EPS).sum(dim=1)
        return lid_estimates
    
class ImplicitLossLIDEstimator(ParametricLIDEstimator):

    def estimate_lid(self, x: torch.Tensor) -> torch.Tensor:
        self.lid_kwargs['norm_mult'] = 0.5
        lid_estimates = self.model.estimate_lid_flipd(x, **self.lid_kwargs)
        return lid_estimates

class FLIPDLIDEstimator(ParametricLIDEstimator):

    def estimate_lid(self, x: torch.Tensor) -> torch.Tensor:
        self.lid_kwargs['norm_mult'] = 1.0
        lid_estimates = self.model.estimate_lid_flipd(x, **self.lid_kwargs)
        return lid_estimates