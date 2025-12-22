from .base import LIDEstimator
import torch
from skdim.id import MLE, TwoNN, ESS
import numpy as np


class NonParametricLIDEstimator(LIDEstimator):

    def estimate_lid(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x_numpy = x.cpu().numpy()
            lid_estimate_numpy = self.method.fit_transform_pw(x_numpy, n_neighbors=self.n_neighbors)
            lid_estimate = torch.from_numpy(lid_estimate_numpy).float().to(x.device)
            return lid_estimate

class MLELIDEstimator(NonParametricLIDEstimator):

    def __init__(self, n_neighbors: int = 100):
        super().__init__()
        self.method = MLE()
        self.n_neighbors = n_neighbors
        
class TwoNNLIDEstimator(NonParametricLIDEstimator):

    def __init__(self, n_neighbors: int = 100):
        super().__init__()
        self.method = TwoNN()
        self.n_neighbors = n_neighbors

class ESSLIDEstimator(NonParametricLIDEstimator):

    def __init__(self, n_neighbors: int = 100):
        super().__init__()
        self.method = ESS()
        self.n_neighbors = n_neighbors