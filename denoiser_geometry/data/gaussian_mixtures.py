from __future__ import annotations
import torch
import torch.utils.data
from typing import Optional, Tuple
from copy import deepcopy

class GaussianMixture:
    def __init__(
        self,
        mu: torch.Tensor,           # (n_comp, ambient_dim)
        sigma: torch.Tensor,        # (n_comp,)
        class_idx: torch.Tensor,    # (n_comp,)
        weight: Optional[torch.Tensor] = None  # (n_comp,) or None
    ):
        """
        Gaussian Mixture Model that stores the analytic probability distribution.
        
        Args:
            mu: Component means of shape (n_comp, ambient_dim)
            sigma: Component standard deviations of shape (n_comp,)
            class_idx: Class indices for each component of shape (n_comp,)
            weight: Component weights of shape (n_comp,). If None, uses uniform weights.
        """
        self.n_comp, self.ambient_dim = mu.shape
        self.mu = mu
        self.sigma = sigma
        self.class_idx = class_idx
        
        # Default to uniform weights if not provided
        if weight is None:
            self.weight = torch.ones(self.n_comp, dtype=torch.float) / self.n_comp
        else:
            # Normalize weights to sum to 1
            self.weight = weight / weight.sum()
        
        # Precompute constants for efficiency
        self.log_weight = torch.log(self.weight)
        self.log_norm_const = -0.5 * self.ambient_dim * torch.log(2 * torch.pi * self.sigma**2)

    def add_noise(self, noise_sigma: float = 0.):
        self.sigma = (self.sigma**2 + noise_sigma**2).sqrt()

    def flow_dist(self, scale: float, sigma: float) -> GaussianMixture:
        flowed_dist = deepcopy(self)
        flowed_dist.mu *= scale
        flowed_dist.sigma *= scale
        flowed_dist.add_noise(sigma)
        return flowed_dist
    
    def to(self, device):
        self.mu = self.mu.to(device)
        self.sigma = self.sigma.to(device)
        self.class_idx = self.class_idx.to(device)
        self.weight = self.weight.to(device)
        self.log_weight = self.log_weight.to(device)
        self.log_norm_const = self.log_norm_const.to(device)
        return self

    
    def state_dict(self) -> dict:
        # Minimal, portable representation
        return {
            "mu": self.mu,
            "sigma": self.sigma,
            "class_idx": self.class_idx,
            "weight": self.weight,  # already normalized in __init__
        }

    @staticmethod
    def from_state_dict(state: dict) -> GaussianMixture:
        return GaussianMixture(
            mu=state["mu"],
            sigma=state["sigma"],
            class_idx=state["class_idx"],
            weight=state.get("weight", None),
        )

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path: str, map_location: Optional[torch.device] = None) -> GaussianMixture:
        state = torch.load(path, map_location=map_location)
        return GaussianMixture.from_state_dict(state)
    
    def sample(self, n_samples: int) -> torch.utils.data.Dataset:
        """
        Sample from the Gaussian mixture model.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            torch.utils.data.Dataset containing (sample, class_label) pairs
        """
        # Sample component assignments using the mixture weights
        component_assignments = torch.multinomial(
            self.weight, 
            n_samples, 
            replacement=True
        )
        
        # Generate samples
        samples = torch.zeros(n_samples, self.ambient_dim, dtype=torch.float, device=self.mu.device)
        labels = torch.zeros(n_samples, dtype=torch.long, device=self.mu.device)
        
        for i in range(n_samples):
            comp_idx = component_assignments[i]
            
            # Sample from the selected Gaussian component
            sample = torch.randn(self.ambient_dim, device=self.mu.device) * self.sigma[comp_idx] + self.mu[comp_idx]
            samples[i] = sample
            
            # Assign the class label for this component
            labels[i] = self.class_idx[comp_idx]
        
        # Create TensorDataset
        dataset = torch.utils.data.TensorDataset(samples, labels)
        
        return dataset
    
    def log_prob(self, x: torch.Tensor, class_conditional: Optional[int] = None) -> torch.Tensor:
        """
        Compute the log probability of samples under the mixture model.
        
        Args:
            x: Input samples of shape (batch_size, ambient_dim)
            class_conditional: If provided, compute class-conditional log probability
            
        Returns:
            Log probabilities of shape (batch_size,)
        """
        batch_size = x.shape[0]
        
        # Compute log probabilities for each component
        # x: (batch_size, ambient_dim), mu: (n_comp, ambient_dim)
        # We need to compute ||x - mu_k||^2 for each component k
        x_expanded = x.unsqueeze(1)  # (batch_size, 1, ambient_dim)
        mu_expanded = self.mu.unsqueeze(0)  # (1, n_comp, ambient_dim)
        
        # Squared distances: (batch_size, n_comp)
        sq_distances = torch.sum((x_expanded - mu_expanded)**2, dim=2)
        
        # Log probabilities for each component: (batch_size, n_comp)
        log_probs_components = (
            self.log_norm_const.unsqueeze(0) - 
            0.5 * sq_distances / (self.sigma**2).unsqueeze(0)
        )
        
        if class_conditional is not None:
            # Filter components that belong to the specified class
            class_mask = (self.class_idx == class_conditional)
            if not class_mask.any():
                raise ValueError(f"No components found for class {class_conditional}")
            
            # Get weights for components of the specified class
            class_weights = self.weight[class_mask]
            class_weights = class_weights / class_weights.sum()  # Renormalize
            
            log_probs_components = log_probs_components[:, class_mask]
            log_weights = torch.log(class_weights).unsqueeze(0)
        else:
            log_weights = self.log_weight.unsqueeze(0)
        
        # Compute log-sum-exp: log(sum(w_k * p_k(x)))
        log_probs_weighted = log_probs_components + log_weights
        log_probs = torch.logsumexp(log_probs_weighted, dim=1)
        
        return log_probs
    
    def score(self, x: torch.Tensor, class_conditional: Optional[int] = None) -> torch.Tensor:
        """
        Compute the score function (gradient of log probability) using autograd.
        
        Args:
            x: Input samples of shape (batch_size, ambient_dim)
            class_conditional: If provided, compute class-conditional score
            
        Returns:
            Score function values of shape (batch_size, ambient_dim)
        """
        x_requires_grad = x.clone().detach().requires_grad_(True)
        log_probs = self.log_prob(x_requires_grad, class_conditional=class_conditional)
        
        # Compute gradients
        scores = torch.autograd.grad(
            outputs=log_probs.sum(),
            inputs=x_requires_grad,
            create_graph=False,
            retain_graph=False
        )[0]
        
        return scores
    
    def score_analytic(self, x: torch.Tensor, class_conditional: Optional[int] = None) -> torch.Tensor:
        """
        Compute the score function analytically.
        
        Args:
            x: Input samples of shape (batch_size, ambient_dim)
            class_conditional: If provided, compute class-conditional score
            
        Returns:
            Score function values of shape (batch_size, ambient_dim)
        """
        batch_size = x.shape[0]
        
        # Compute component responsibilities (posterior probabilities)
        x_expanded = x.unsqueeze(1)  # (batch_size, 1, ambient_dim)
        mu_expanded = self.mu.unsqueeze(0)  # (1, n_comp, ambient_dim)
        
        # Squared distances: (batch_size, n_comp)
        sq_distances = torch.sum((x_expanded - mu_expanded)**2, dim=2)
        
        # Log probabilities for each component: (batch_size, n_comp)
        log_probs_components = (
            self.log_norm_const.unsqueeze(0) - 
            0.5 * sq_distances / (self.sigma**2).unsqueeze(0)
        )
        
        if class_conditional is not None:
            # Filter components that belong to the specified class
            class_mask = (self.class_idx == class_conditional)
            if not class_mask.any():
                raise ValueError(f"No components found for class {class_conditional}")
            
            class_weights = self.weight[class_mask]
            class_weights = class_weights / class_weights.sum()
            
            log_probs_components = log_probs_components[:, class_mask]
            log_weights = torch.log(class_weights).unsqueeze(0)
            mu_filtered = self.mu[class_mask]
            sigma_filtered = self.sigma[class_mask]
        else:
            log_weights = self.log_weight.unsqueeze(0)
            mu_filtered = self.mu
            sigma_filtered = self.sigma
        
        # Weighted log probabilities: (batch_size, n_comp)
        log_probs_weighted = log_probs_components + log_weights
        
        # Component responsibilities: (batch_size, n_comp)
        log_responsibilities = log_probs_weighted - torch.logsumexp(log_probs_weighted, dim=1, keepdim=True)
        responsibilities = torch.exp(log_responsibilities)
        
        # Compute score analytically
        # For each component: responsibility * (mu - x) / sigma^2
        mu_filtered_expanded = mu_filtered.unsqueeze(0)  # (1, n_comp, ambient_dim)
        
        # Component-wise scores: (batch_size, n_comp, ambient_dim)
        component_scores = (
            responsibilities.unsqueeze(2) * 
            (mu_filtered_expanded - x_expanded[:, :, :mu_filtered.shape[1]]) / 
            (sigma_filtered**2).unsqueeze(0).unsqueeze(2)
        )
        
        # Sum over components: (batch_size, ambient_dim)
        scores = torch.sum(component_scores, dim=1)
        
        return scores


def get_bimodal_set(dim: int = 1) -> GaussianMixture:
    """Create a bimodal Gaussian mixture."""
    mu = torch.cat([
        torch.ones(1, dim),
        torch.ones(1, dim) * -1.
    ], dim=0)
    sigma = torch.ones(2) * 0.1
    class_idx = torch.tensor([0, 1])
    
    return GaussianMixture(mu=mu, sigma=sigma, class_idx=class_idx)

def get_trimodal_set(dim: int = 1) -> GaussianMixture:
    """Create a trimodal Gaussian mixture."""
    mu = torch.cat([
        torch.ones(1, dim),
        torch.zeros(1, dim),
        torch.ones(1, dim) * -1.
    ], dim=0)
    sigma = torch.ones(3) * 0.1
    class_idx = torch.tensor([0, 1, 2])
    
    return GaussianMixture(mu=mu, sigma=sigma, class_idx=class_idx)
