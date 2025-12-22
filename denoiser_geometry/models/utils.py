import torch
import torch.nn as nn
import math


class FourierFeatures(nn.Module):
    """Fourier feature embedding for time conditioning."""
    
    def __init__(self, embed_dim, max_freq=10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_freq = max_freq
        
        # Create frequency bands
        freqs = torch.exp(torch.linspace(0, math.log(max_freq), embed_dim // 2))
        self.register_buffer('freqs', freqs)
    
    def forward(self, t):
        """
        Args:
            t: Time tensor of shape (batch_size,)
        Returns:
            Fourier features of shape (batch_size, embed_dim)
        """
        # t shape: (batch_size,) -> (batch_size, 1)
        t = t.unsqueeze(-1)
        
        # Compute sin and cos features
        args = t * self.freqs.unsqueeze(0)  # (batch_size, embed_dim//2)
        sin_features = torch.sin(args)
        cos_features = torch.cos(args)
        
        # Concatenate sin and cos features
        return torch.cat([sin_features, cos_features], dim=-1)  # (batch_size, embed_dim)