import torch
import torch.nn as nn
from .utils import FourierFeatures
from .base import BaseDenoiserModel

class DiffusionMLPKamkari(BaseDenoiserModel):
    """MLP-based diffusion model with bottleneck architecture and skip connections."""
    
    def __init__(
        self,
        ambient_dim,
        num_classes,
        hidden_dims=[512, 256, 128, 256, 512],  # Example bottleneck: decreases then increases
        time_embed_dim=128,
        class_embed_dim=128,
        dropout=0.1,
        activation='silu',
        use_layer_norm=False
    ):
        super().__init__(ambient_dim, num_classes)
        self.time_embed_dim = time_embed_dim
        self.class_embed_dim = class_embed_dim
        
        # Validate bottleneck architecture
        if len(hidden_dims) % 2 == 0:
            raise ValueError("hidden_dims must have odd length to form a bottleneck")
        
        self.L = len(hidden_dims) // 2  # Number of layers to bottleneck
        self.hidden_dims = hidden_dims
        
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
        
        # Build bottleneck MLP with skip connections
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        self.dropouts = nn.ModuleList() if dropout > 0 else None
        
        input_dim = ambient_dim + conditioning_dim
        
        # First half: encoder (down to bottleneck)
        for i in range(self.L + 1):  # 0 to L (inclusive)
            if i == 0:
                layer_input_dim = input_dim
            else:
                layer_input_dim = self.hidden_dims[i - 1]
            
            layer_output_dim = self.hidden_dims[i]
            self.layers.append(nn.Linear(layer_input_dim, layer_output_dim))
            
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(layer_output_dim))
            if dropout > 0:
                self.dropouts.append(nn.Dropout(dropout))
        
        # Second half: decoder (up from bottleneck) with skip connections
        for i in range(self.L + 1, 2 * self.L + 1):  # L+1 to 2L (inclusive)
            # Skip connection from corresponding encoder layer
            skip_layer_idx = 2 * self.L - i  # This gives us the mirrored layer index
            
            if skip_layer_idx == 0:
                # Skip from input
                skip_dim = input_dim
            else:
                skip_dim = self.hidden_dims[skip_layer_idx - 1]
            
            # Input from previous layer + skip connection
            layer_input_dim = self.hidden_dims[i - 1] + skip_dim
            layer_output_dim = self.hidden_dims[i]
            
            self.layers.append(nn.Linear(layer_input_dim, layer_output_dim))
            
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(layer_output_dim))
            if dropout > 0:
                self.dropouts.append(nn.Dropout(dropout))
        
        # Output layer (from last hidden layer to ambient_dim)
        final_skip_dim = input_dim  # Skip connection from input
        final_input_dim = self.hidden_dims[-1] + final_skip_dim
        self.output_layer = nn.Linear(final_input_dim, ambient_dim)
        
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
        Predict noise added to x with class conditioning using bottleneck architecture.
        
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
        tc_embed = torch.cat([t_embed, c_embed], dim=-1)
        tc_embed = self.conditioning_proj(tc_embed)
        
        # Concatenate x and conditioning embedding
        xtc = torch.cat([x, tc_embed], dim=-1)
        
        # Store activations for skip connections
        activations = [xtc]  # Index 0: input
        
        # Forward pass through encoder (first L+1 layers)
        current = xtc
        for i in range(self.L + 1):
            current = self.layers[i](current)
            
            if self.layer_norms is not None:
                current = self.layer_norms[i](current)
            
            current = self.activation(current)
            
            if self.dropouts is not None:
                current = self.dropouts[i](current)
            
            activations.append(current)
        
        # Forward pass through decoder (remaining L layers) with skip connections
        for i in range(self.L + 1, 2 * self.L + 1):
            # Get skip connection from mirrored encoder layer
            skip_idx = 2 * self.L - i
            skip_activation = activations[skip_idx]
            
            # Concatenate current activation with skip connection
            current = torch.cat([current, skip_activation], dim=-1)
            current = self.layers[i](current)
            
            if self.layer_norms is not None:
                current = self.layer_norms[i](current)
            
            current = self.activation(current)
            
            if self.dropouts is not None:
                current = self.dropouts[i](current)
        
        # Final output layer with skip connection from input
        current = torch.cat([current, xtc], dim=-1)
        noise_pred = self.output_layer(current)
        
        return noise_pred