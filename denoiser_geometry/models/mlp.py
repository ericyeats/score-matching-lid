import torch
import torch.nn as nn
from .utils import FourierFeatures
from .base import BaseDenoiserModel


class DiffusionMLP(BaseDenoiserModel):
    """MLP-based diffusion model that predicts noise from diffused input with class conditioning."""
    
    def __init__(
        self,
        ambient_dim,
        num_classes,
        hidden_dims=[512, 512, 512],
        time_embed_dim=128,
        class_embed_dim=128,
        dropout=0.1,
        activation='silu',
        use_layer_norm=False
    ):
        super().__init__(ambient_dim, num_classes)
        self.time_embed_dim = time_embed_dim
        self.class_embed_dim = class_embed_dim
        
        # Time embedding
        self.time_embed = FourierFeatures(time_embed_dim)
        
        # Class embedding
        self.class_embed = nn.Embedding(num_classes + 1, class_embed_dim)
        
        # Combined conditioning embedding
        conditioning_dim = time_embed_dim + class_embed_dim
        self.conditioning_proj = nn.Sequential(
            nn.Linear(conditioning_dim, conditioning_dim),
            nn.SiLU(),
            nn.Linear(conditioning_dim, conditioning_dim)
        )
        
        # Activation function
        if activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build MLP layers
        layers = []
        input_dim = ambient_dim + conditioning_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        # Output layer (predicts noise)
        layers.append(nn.Linear(input_dim, ambient_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, x, t, class_idx):
        """
        Predict noise added to x with class conditioning.
        
        Args:
            x: Diffused input of shape (batch_size, ambient_dim)
            t: Diffusion time of shape (batch_size,) with values in [0, 1]
            class_idx: Class indices of shape (batch_size,) with values in [0, num_classes)
            
        Returns:
            Predicted noise of shape (batch_size, ambient_dim)
        """
        # Embed time
        t_embed = self.time_embed(t)  # (batch_size, time_embed_dim)
        
        # Embed class
        c_embed = self.class_embed(class_idx)  # (batch_size, class_embed_dim)
        
        # Combine time and class embeddings
        tc_embed = torch.cat([t_embed, c_embed], dim=-1)  # (batch_size, time_embed_dim + class_embed_dim)
        tc_embed = self.conditioning_proj(tc_embed)  # Process combined embedding
        
        # Concatenate x and conditioning embedding
        xtc = torch.cat([x, tc_embed], dim=-1)  # (batch_size, ambient_dim + conditioning_dim)
        
        # Pass through network
        noise_pred = self.network(xtc)
        
        return noise_pred