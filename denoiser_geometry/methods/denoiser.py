# denoiser - this generalizes both diffusion generative models and rectified flows
# should implement the interface for calculating LID, Tangent & Normal spaces, etc

import torch
from ..models.base import BaseDenoiserModel
from copy import deepcopy
from typing import Callable
from functools import partial

def duplicate_single(tensor, n_tiles):
    return tensor.repeat_interleave(n_tiles, dim=0)

def unit_vec(_x: torch.Tensor) -> torch.Tensor:
    _x_norm = _x.view(_x.shape[0], -1).norm(p=2, dim=1)
    _x_dim = _x.dim()
    for i in range(_x_dim - 1):
        _x_norm = _x_norm.unsqueeze(-1)
    return _x / _x_norm

def inner_prod(a: torch.Tensor, b: torch.Tensor):
    assert a.shape == b.shape
    return (a*b).sum(dim=list(range(1, a.dim())))

            
def skilling_hutchinson_jacobian_trace_estimate(
    func: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    noise_type='rademacher',
    noise: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    
    if not isinstance(noise, torch.Tensor):
        if noise_type == 'rademacher':
            noise = torch.randint_like(x, 2).float() * 2. - 1.
        elif noise_type == 'gaussian':
            noise = torch.randn_like(x)
        else:
            raise ValueError(f"noise type {noise_type} not recognized")

    with torch.enable_grad():
        x.requires_grad_(True)
        y, vjp = torch.autograd.functional.vjp(func, x, noise)
        x.requires_grad_(False)
    
    return y, inner_prod(vjp, noise)

def mask_with_rate(tensor, rate, mask_with=-1):
    """
    Randomly set elements to -1 with given probability.
    
    Args:
        tensor: Tensor of integers
        rate: Float in [0, 1], probability of setting each element to -1
        
    Returns:
        Tensor with elements randomly set to -1
    """
    mask = torch.rand_like(tensor, dtype=torch.float) < rate
    result = tensor.clone()
    result[mask] = mask_with
    return result

def duplicate(x: torch.Tensor, t: torch.Tensor, class_idx: torch.Tensor, n_samples: int | torch.Tensor = 1):

    if isinstance(n_samples, int):
        noise = torch.randn(n_samples, x.shape[1], device=x.device)
    else:
        noise = n_samples
        n_samples = noise.shape[0]

    # duplicate x, t, class_idx by n_samples
    x_dup = duplicate_single(x, n_samples)
    t_dup = duplicate_single(t, n_samples)
    class_idx_dup = duplicate_single(class_idx, n_samples)
    noise_dup = noise.repeat(x.shape[0], 1)

    return x_dup, t_dup, class_idx_dup, noise_dup

def preprocess(
        x: torch.Tensor, 
        t: torch.Tensor | float, 
        class_idx: int | torch.Tensor = -1,
    ):

    ## ensure t is Tensor and shape (batch_size,)
    if isinstance(t, float):
        t = torch.ones((x.shape[0],), device=x.device) * t

    assert isinstance(t, torch.Tensor)

    if t.dim() == 0:
        t = t.unsqueeze(0)

    if t.shape[0] != x.shape[0]:
        t = t.expand(x.shape[0]) # ensure that t is (batch_size,)

    t = t.to(x.device)

    ## convert class_idx to Tensor
    if isinstance(class_idx, int):
        class_idx = torch.full((x.shape[0],), fill_value=class_idx, dtype=int, device=x.device)
    
    return x, t, class_idx


class Denoiser(torch.nn.Module):
    """The base class for Diffusion and Rectified Flow models
    Denoisers operated on x in X and t in [0, 1].
    By default, they will use the OT Gaussian paths for add_noise.
    Moreover, following diffusion literature, we will use the convention 
    t==0 -> data_dist
    t==1 -> noise_dist
    """

    def __init__(
        self,
        model: BaseDenoiserModel,
        ambient_dim: int,
        **kwargs
    ):
        super().__init__()
        self.model = model
        self.ambient_dim = ambient_dim

    def pred_eps(self, x: torch.Tensor, t: torch.Tensor, class_idx: int | torch.Tensor = -1) -> torch.Tensor:
        """Predict the noise used to perturb a sample
        This may need to be altered for different methods/parameterizations"""
        x, t, class_idx = preprocess(x, t, class_idx)
        return self.model(x, t, class_idx) # returns output in the shape of x
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, class_idx: int | torch.Tensor = -1) -> torch.Tensor:
        return self.pred_eps(x, t, class_idx)
    
    def get_a_b(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a (data scale) and b (noise scale) from t
        This is OT Gaussian paths by default, but this can and should be re-defined"""
        return 1. - t, t
    
    def pred_score(self, x: torch.Tensor, t: torch.Tensor, class_idx: int | torch.Tensor = -1) -> torch.Tensor:
        x, t, class_idx = preprocess(x, t, class_idx)
        a, b = self.get_a_b(t)
        return -self.pred_eps(x, t, class_idx) / b
    
    def add_noise(
            self, 
            x: torch.Tensor, 
            t: torch.Tensor | float, 
            noise: torch.Tensor | None = None
        ) -> tuple[torch.Tensor, torch.Tensor]:
        "Add noise to the x samples"
        x, t, _ = preprocess(x, t)

        a, b = self.get_a_b(t)

        if noise is None:
            noise = torch.randn_like(x)

        x_noisy = a[:, None] * x + b[:, None] * noise

        return x_noisy, noise

    def training_loss(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        class_idx: int | torch.Tensor = -1,
        noise: torch.Tensor | None = None,
        reduce: bool = True
    ) -> torch.Tensor:
        "The training loss for a sample x and tensor t."

        # create noise if necessary
        if noise is None:
            noise = torch.randn_like(x)
        
        x, t, class_idx = preprocess(x, t, class_idx)
        x_noisy, noise = self.add_noise(x, t, noise)

        eps = self.pred_eps(x_noisy, t, class_idx)

        loss = (eps - noise).square()

        if reduce:
            loss = loss.mean()
        
        return loss
    
    def estimate_lid_subspace(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        class_idx: int | torch.Tensor = -1,
        n_samples: int | torch.Tensor = 100,
        estimate_tangent_subspace: bool = False
    ) -> torch.Tensor:
        """Estimate the LID of the samples at time t
        Additionally estimate the tangent subspace, if desired"""

        with torch.no_grad():
            self.model.eval()
            # noise = torch.randn((n_samples, x.shape[1]), device=x.device)
            x, t, class_idx = preprocess(x, t, class_idx)
            x_dup, t_dup, class_idx_dup, noise_dup = duplicate(x, t, class_idx, n_samples)

            if isinstance(n_samples, torch.Tensor):
                n_samples = n_samples.shape[0]

            losses = self.training_loss(
                x_dup, t_dup, class_idx_dup, noise_dup, reduce=False
            ).view(x.shape[0], n_samples, x.shape[1])
            # losses is (batch_size, n_samples, ambient_dim)

            lids = losses.sum(dim=-1).mean(dim=1) # (batch_size,)

            if not estimate_tangent_subspace:
                return lids

            ## calculate the tangent subspace too
            covariance_matrices = torch.bmm(losses.transpose(1,2), losses) / n_samples # (batch_size, ambient_dim, ambient_dim)
            eigvals, eigvecs = torch.linalg.eigh(covariance_matrices)

            return lids, eigvals, eigvecs

    def estimate_lid_denoising_loss(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        class_idx: int | torch.Tensor = -1,
        n_samples: int | torch.Tensor = 100
    ) -> torch.Tensor:
        """Estimate the LID of the samples at time t
        Additionally estimate the tangent subspace, if desired"""

        with torch.no_grad():
            self.model.eval()
            # noise = torch.randn((n_samples, x.shape[1]), device=x.device)
            x, t, class_idx = preprocess(x, t, class_idx)
            
            loss = 0

            if isinstance(n_samples, torch.Tensor):
                noise = n_samples
                n_samples = n_samples.shape[0]
            else:
                noise = torch.randn((n_samples, x.shape[1]), device=x.device)

            for i in range(n_samples):
                _noise = noise[i][None, :].expand_as(x)
                loss += self.training_loss(
                    x, t, class_idx, _noise, reduce=False
                ).sum(dim=1)
            # losses is (batch_size, n_samples, ambient_dim)

            lids = loss / n_samples

            return lids
        
    def estimate_lid_flipd(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        class_idx: int | torch.Tensor = -1,
        norm_mult: float = 1.,
        n_samples: int | torch.Tensor = 8, # number of Rademacher samples or Rademacher instances
    ) -> torch.Tensor:
        # estimate the LID using the FLIPD estimator
        # Kamkari et al 2024
        # LID(x) = D + deriv_t log p(x)

        if isinstance(n_samples, torch.Tensor):
            noise = n_samples
            n_samples = n_samples.shape[0]
        else:
            noise = torch.randint(0, 2, (n_samples, x.shape[1]), device=x.device) * 2. - 1.

        with torch.no_grad():
            self.model.eval()
            x, t, class_idx = preprocess(x, t, class_idx)
            # get the scale of the data
            a, b = self.get_a_b(t)
            x_scl = a[:, None] * x
            f = partial(self.pred_eps, t=t, class_idx=class_idx)
            trace_est = 0.
            for i in range(n_samples):
                _noise = noise[i][None, :].expand_as(x)
                y, trace = skilling_hutchinson_jacobian_trace_estimate(f, x_scl, noise=_noise)
                # computed divergence of eps pred: this is -sigma*score
                trace = -b * trace
                trace_est += trace
            trace_est /= n_samples
            eps_norm = inner_prod(y, y) # y is same for deterministic network
            return self.model.ambient_dim + (trace_est + norm_mult*eps_norm)
    
    def train_dsm(
        self,
        train_data: torch.utils.data.Dataset,
        batch_size: int = 100,
        n_batches: int = 1000,
        learning_rate: float = 1e-3,
        class_drop_rate: float = 0.3,
        device: str = 'cpu',
        print_every: int = 100,
        # Cosine annealing parameters
        use_cosine_annealing: bool = True,
        min_lr: float = 1e-5,
        warmup_batches: int = 0,
        time_range: list[float] = [0., 1.],
        grad_clip: bool = True,
        **opt_kwargs
    ):
        # train_data is a dataset of n_data_samples x ambient_dim
        # train on n_batches of size batch_size, sampled with replacement
        
        # Setup optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, **opt_kwargs)
        
        # Setup cosine annealing scheduler
        if use_cosine_annealing:
            # CosineAnnealingLR goes from current LR to eta_min over T_max steps
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=n_batches - warmup_batches,  # Total steps for annealing
                eta_min=min_lr
            )
            
            # Optional: Add warmup scheduler
            if warmup_batches > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.1,  # Start at 10% of base LR
                    end_factor=1.0,    # End at 100% of base LR
                    total_iters=warmup_batches
                )
        
        # Create sampler for sampling with replacement
        sampler = torch.utils.data.RandomSampler(
            train_data, 
            replacement=True, 
            num_samples=n_batches * batch_size
        )
        # Create data loader with the sampler
        dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True
        )
        
        self.model.train()
        running_loss = 0.0

        if isinstance(self.model, torch.nn.DataParallel):
            num_classes = self.model.module.num_classes
        else:
            num_classes = self.model.num_classes
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= n_batches:
                break
                
            # Get batch of data
            if isinstance(batch, (tuple, list)):
                # If dataset returns (data, labels, lids)
                x0, class_idx = batch
                x0 = x0.to(device)
                class_idx = class_idx.to(device)
                class_idx = mask_with_rate(class_idx, class_drop_rate, mask_with=num_classes)
            else:
                # If dataset only returns data
                x0 = batch.to(device)
                class_idx = torch.full(
                    (x0.shape[0],), fill_value=num_classes, dtype=int, device=device
                )
            
            # Sample random times uniformly in time_range
            t = torch.rand(batch_size, device=device) 
            t = t * (time_range[1] - time_range[0]) + time_range[0]
            loss = self.training_loss(x0, t, class_idx)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            if grad_clip: torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.)
            optimizer.step()
            
            # Update learning rate with cosine annealing
            if use_cosine_annealing:
                if warmup_batches > 0 and batch_idx < warmup_batches:
                    # During warmup phase
                    warmup_scheduler.step()
                elif batch_idx >= warmup_batches:
                    # During cosine annealing phase
                    scheduler.step()
            
            # Track loss
            running_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % print_every == 0:
                avg_loss = running_loss / print_every
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Batch {batch_idx + 1}/{n_batches}, Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")
                running_loss = 0.0
            
        print("Training completed!")

    def unlearn_esd(
        self,
        forget_class: int,
        n_samples: int = 200,
        eta: float = 1.,
        batch_size: int = 100,
        n_batches: int = 1000,
        learning_rate: float = 1e-4,
        device: str = 'cuda',
        print_every: int = 100,
        # Cosine annealing parameters
        use_cosine_annealing: bool = True,
        min_lr: float = 1e-6,
        warmup_batches: int = 0,
        **opt_kwargs
    ):
        "Implementation of unlearning with ESD https://openaccess.thecvf.com/content/ICCV2023/html/Gandikota_Erasing_Concepts_from_Diffusion_Models_ICCV_2023_paper.html"
        # train_data is a dataset of n_data_samples x ambient_dim
        # train on n_batches of size batch_size, sampled with replacement

        train_data = self.deterministic_sample(n_samples, forget_class, device=device)
        labels = torch.full((n_samples,), fill_value=forget_class, dtype=torch.int, device=device)
        train_data = torch.utils.data.TensorDataset(train_data, labels)

        original_model = deepcopy(self)

        def esd_loss(
            x: torch.Tensor, 
            t: torch.Tensor, 
            class_idx: int | torch.Tensor = -1,
            noise: torch.Tensor | None = None,
            reduce: bool = True
        ) -> torch.Tensor:

            # create noise if necessary
            if noise is None:
                noise = torch.randn_like(x)
            
            x, t, class_idx = preprocess(x, t, class_idx)
            x_noisy, noise = self.add_noise(x, t, noise)

            eps = self.pred_eps(x_noisy, t, class_idx)

            with torch.no_grad():
                eps_orig_uncond = original_model.pred_eps(x_noisy, t, original_model.model.num_classes)
                eps_orig_cond = original_model.pred_eps(x_noisy, t, forget_class)
                eps_target = eps_orig_uncond - eta * (eps_orig_cond - eps_orig_uncond)

            loss = (eps - eps_target).square()

            if reduce:
                loss = loss.mean()
            
            return loss
        
        # Setup optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, **opt_kwargs)
        
        # Setup cosine annealing scheduler
        if use_cosine_annealing:
            # CosineAnnealingLR goes from current LR to eta_min over T_max steps
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=n_batches - warmup_batches,  # Total steps for annealing
                eta_min=min_lr
            )
            
            # Optional: Add warmup scheduler
            if warmup_batches > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.1,  # Start at 10% of base LR
                    end_factor=1.0,    # End at 100% of base LR
                    total_iters=warmup_batches
                )
        
        # Create sampler for sampling with replacement
        sampler = torch.utils.data.RandomSampler(
            train_data, 
            replacement=True, 
            num_samples=n_batches * batch_size
        )
        # Create data loader with the sampler
        dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True
        )
        
        self.model.train()
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= n_batches:
                break
                
            # Get batch of data
            if isinstance(batch, (tuple, list)):
                # If dataset returns (data, labels)
                x0, class_idx = batch
                x0 = x0.to(device)
                class_idx = class_idx.to(device)
            else:
                # If dataset only returns data
                x0 = batch.to(device)
                class_idx = torch.full((x0.shape[0],), fill_value=0, dtype=int)
            
            # Sample random times uniformly from [0, 1]
            t = torch.rand(batch_size, device=device)
            loss = esd_loss(x0, t, class_idx)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update learning rate with cosine annealing
            if use_cosine_annealing:
                if warmup_batches > 0 and batch_idx < warmup_batches:
                    # During warmup phase
                    warmup_scheduler.step()
                elif batch_idx >= warmup_batches:
                    # During cosine annealing phase
                    scheduler.step()
            
            # Track loss
            running_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % print_every == 0:
                avg_loss = running_loss / print_every
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Batch {batch_idx + 1}/{n_batches}, Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")
                running_loss = 0.0
            
        print("Training completed!")
