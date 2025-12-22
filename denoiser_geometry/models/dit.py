import torch
import torch.nn as nn
from .utils import FourierFeatures
from .base import BaseDenoiserModel

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: Input of shape (batch_size, seq_len, embed_dim)
        Returns:
            Output of shape (batch_size, seq_len, embed_dim)
        """
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, num_heads, N, head_dim)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP."""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input of shape (batch_size, seq_len, embed_dim)
        """
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class DiffusionTransformer(BaseDenoiserModel):
    """Transformer-based diffusion model that predicts noise from diffused input with class conditioning."""
    
    def __init__(
        self,
        ambient_dim,
        num_classes,
        embed_dim=512,
        num_layers=6,
        num_heads=8,
        mlp_ratio=4.0,
        time_embed_dim=128,
        class_embed_dim=128,
        patch_size=None,
        dropout=0.1
    ):
        super().__init__(ambient_dim, num_classes)
        self.embed_dim = embed_dim
        self.patch_size = patch_size or ambient_dim  # Default: treat whole vector as one patch
        
        # Calculate number of patches
        assert ambient_dim % self.patch_size == 0, "ambient_dim must be divisible by patch_size"
        self.num_patches = ambient_dim // self.patch_size
        
        # Time embedding
        self.time_embed = FourierFeatures(time_embed_dim)
        self.time_proj = nn.Linear(time_embed_dim, embed_dim)
        
        # Class embedding
        self.class_embed = nn.Embedding(num_classes + 1, class_embed_dim)
        self.class_proj = nn.Linear(class_embed_dim, embed_dim)
        
        # Combined conditioning projection
        self.conditioning_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Patch embedding
        self.patch_embed = nn.Linear(self.patch_size, embed_dim)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, self.patch_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        
        # Initialize output layer to zero (good practice for diffusion models)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    
    def patchify(self, x):
        """
        Convert input to patches.
        
        Args:
            x: Input of shape (batch_size, ambient_dim)
        Returns:
            Patches of shape (batch_size, num_patches, patch_size)
        """
        B = x.shape[0]
        return x.reshape(B, self.num_patches, self.patch_size)
    
    def unpatchify(self, x):
        """
        Convert patches back to original shape.
        
        Args:
            x: Patches of shape (batch_size, num_patches, patch_size)
        Returns:
            Output of shape (batch_size, ambient_dim)
        """
        B = x.shape[0]
        return x.reshape(B, self.ambient_dim)
    
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
        B = x.shape[0]
        
        # Embed time and project to embed_dim
        t_embed = self.time_embed(t)  # (B, time_embed_dim)
        t_embed = self.time_proj(t_embed)  # (B, embed_dim)
        
        # Embed class and project to embed_dim
        c_embed = self.class_embed(class_idx)  # (B, class_embed_dim)
        c_embed = self.class_proj(c_embed)  # (B, embed_dim)
        
        # Combine time and class embeddings
        tc_embed = torch.cat([t_embed, c_embed], dim=-1)  # (B, embed_dim * 2)
        tc_embed = self.conditioning_proj(tc_embed)  # (B, embed_dim)
        
        # Patchify input
        x_patches = self.patchify(x)  # (B, num_patches, patch_size)
        
        # Patch embedding
        x_embed = self.patch_embed(x_patches)  # (B, num_patches, embed_dim)
        
        # Add positional embeddings
        x_embed = x_embed + self.pos_embed
        
        # Add combined conditioning embedding to each patch (broadcast)
        x_embed = x_embed + tc_embed.unsqueeze(1)  # (B, num_patches, embed_dim)
        
        # Apply transformer blocks
        for block in self.blocks:
            x_embed = block(x_embed)
        
        # Final norm and output projection
        x_embed = self.norm(x_embed)
        noise_pred = self.head(x_embed)  # (B, num_patches, patch_size)
        
        # Unpatchify to get final output
        noise_pred = self.unpatchify(noise_pred)  # (B, ambient_dim)
        
        return noise_pred