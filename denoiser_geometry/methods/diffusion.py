# diffusion

from .denoiser import Denoiser, preprocess
from .sampled import Sampled
import torch

BETA_MIN = 0.1
BETA_MAX = 20.

def get_beta_t(t: torch.Tensor) -> torch.Tensor:
    # assume t in [0, 1]
    return (BETA_MAX - BETA_MIN) * t + BETA_MIN

def get_beta_t_int(t: torch.Tensor) -> torch.Tensor:
    # assume t in 0, 1
    return 0.5 * t.square() * (BETA_MAX - BETA_MIN) + t * BETA_MIN

class Diffusion(Denoiser, Sampled):
    
    def get_ambient_dim(self) -> int:
        return self.model.ambient_dim

    def get_num_classes(self):
        return self.model.num_classes

    def get_a_b(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a (data scale) and b (noise scale) from t"""

        
        log_scale = -1. * get_beta_t_int(t)
        data_scale = (log_scale  / 2).exp()
        noise_scale = (1. - (log_scale).exp()).sqrt()
        return data_scale, noise_scale
    
    def velocity(self, x, t, class_idx):
        "Velocity as the Prob Flow ODE (Song et al 2021)"

        # FORWARD
        # dx = f(x,t) dt + g(t) dW

        # REVERSE
        # dx = [f(x,t) - (1/2)g(t)^2 score(x, t)] dt

        # f(x,t) = -(1/2) b(t) x
        # g(t) = b(t)^(1/2)

        a, b = self.get_a_b(t)

        eps = self.pred_eps(x, t, class_idx)
        sxt = -eps / b[:, None]

        beta_t = get_beta_t(t)

        fxt = -0.5 * beta_t[:, None] * x

        gt2 = beta_t[:, None]

        dx_dt = fxt - 0.5 * gt2 * sxt

        return dx_dt


        
    
