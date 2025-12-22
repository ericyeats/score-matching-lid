from .denoiser import Denoiser, preprocess
from .sampled import Sampled
import torch

class FlowMatch(Denoiser, Sampled):

    def get_ambient_dim(self) -> int:
        return self.model.ambient_dim

    def get_num_classes(self):
        return self.model.num_classes

    def pred_eps(self, x: torch.Tensor, t: torch.Tensor, class_idx: int | torch.Tensor = -1) -> torch.Tensor:
        """Predict the noise used to perturb a sample
        This may need to be altered for different methods/parameterizations"""
        x, t, class_idx = preprocess(x, t, class_idx)
        a, b = self.get_a_b(t)

        # parameterize according to Esser et al 2024

        return a[:, None] * self.model(x, t, class_idx) + x
    
    def velocity(self, x, t, class_idx):

        vel = self.model(x, t, class_idx)

        return vel
    
