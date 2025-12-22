import torch

class Sampled:

    def __init__(self, *args, **kwargs):
        super().__init__()
        # do nothing

    def get_num_classes(self) -> int:
        raise NotImplementedError("Get Num classes not yet implemented")

    def get_ambient_dim(self) -> int:
        raise NotImplementedError("get_ambient_dim not yet implemented")

    def velocity(self, x: torch.Tensor, t: torch.Tensor, class_idx: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Velocity not yet implemented")
    
    def deterministic_sample(
            self,
            n_samples: int,
            class_idx: torch.Tensor | int,
            n_steps: int = 100,
            device: str = 'cpu',
            clf_guidance: float = 0.
            ) -> torch.Tensor:

        if hasattr(self, "model") and isinstance(self.model, torch.nn.Module):
            self.model.eval()

        with torch.no_grad():

            # Start from Gaussian noise at t=1
            x = torch.randn(n_samples, self.get_ambient_dim(), device=device)

            if isinstance(class_idx, int):
                class_idx = torch.full((n_samples,), fill_value=class_idx, dtype=int, device=device)

            # Ensure class_idx has the right shape and device
            if class_idx.shape[0] != n_samples:
                class_idx = class_idx.repeat(n_samples)
            class_idx = class_idx.to(device)

            dt = 1.0 / n_steps

            # Integrate ODE from t=1 to t=0 using Euler method
            for i in range(n_steps):
                t = torch.full((n_samples,), 1.0 - i * dt, device=device)
                
                with torch.no_grad():
                    # Get velocity field v(x_t, t)
                    velocity = self.velocity(x, t, class_idx)

                    if clf_guidance > 0.:
                        uncond_labels = torch.full((n_samples,), fill_value=self.get_num_classes(), dtype=int, device=device)
                        unc_velocity = self.velocity(x, t, uncond_labels)
                        velocity = unc_velocity + clf_guidance * (velocity - unc_velocity)
                    
                    # Euler step: x_{t-dt} = x_t + v(x_t, t) * -dt
                    x = x + velocity * (-dt)

            return x