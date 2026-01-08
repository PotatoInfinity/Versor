import torch
import torch.nn as nn
from .layers import GeometricAttention, GeometricLinear
from .core import normalize_cl41, gp_cl41, reverse_cl41, inner_cl41

class GeometricActivation(nn.Module):
    """
    Magnitude-based activation function for multivectors.
    Preserves the 'direction' in GA space while applying a non-linear gate
    based on the scalar magnitude (norm).
    """
    def forward(self, x):
        # Using the inner product to find the magnitude squared
        norm_sq = inner_cl41(x, x)
        norm = torch.sqrt(torch.abs(norm_sq) + 1e-8)
        # ReLU gate on the magnitude
        gate = torch.relu(norm) / (norm + 1e-8)
        return x * gate.unsqueeze(-1)

class GeometricBlock(nn.Module):
    """
    A single Transformer block specialized for GA.
    Includes GPA (Geometric Product Attention) and a Geometric MLP.
    """
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.attn = GeometricAttention(embed_dim, n_heads)
        # LayerNorm is applied across the dimension and multivector lanes
        self.ln1 = nn.LayerNorm([embed_dim, 32])
        self.ln2 = nn.LayerNorm([embed_dim, 32])
        
        self.mlp = nn.Sequential(
            GeometricLinear(embed_dim, 4 * embed_dim),
            GeometricActivation(),
            GeometricLinear(4 * embed_dim, embed_dim)
        )
        
    def forward(self, x):
        # Residual Connection + Attention
        x = x + self.attn(self.ln1(x))
        x = normalize_cl41(x) # Manifold projection
        
        # Residual Connection + MLP
        x = x + self.mlp(self.ln2(x))
        x = normalize_cl41(x)
        return x

class RecursiveRotorAccumulator(nn.Module):
    """
    Stabilized Cumulative Rotor Pooling.
    
    Instead of a slow Python loop, this uses a vectorized 'state-sum' 
    approach which is more stable for long sequences (N=64).
    The sequence is transformed into a 'Geometric Stream' that is
    normalized to prevent energy explosion.
    """
    def __init__(self, embed_dim):
        super().__init__()
        # Project each time step to a 'delta-rotor'
        self.mixer = GeometricLinear(embed_dim, embed_dim)
        
    def forward(self, x):
        # x: (B, S, D, 32)
        
        # 1. Transform sequence to delta-rotors
        delta = self.mixer(x) 
        
        # 2. Cumulative Mean (Global Context)
        # We use a stable mean across the time dimension
        # This acts as a 'Mean Rotor' which is more stable than recursive sandwiching
        psi = delta.mean(dim=1) 
        
        # 3. Final Manifold Stability Gate
        # Hard-clamping the multivector components to prevent logit explosion
        psi = torch.tanh(psi) 
        return normalize_cl41(psi)

class GeometricTransformer(nn.Module):
    """
    Full Geometric Transformer for Conformal Geometric Algebra Cl(4,1).
    Equipped with Geometric Blocks and optional Stabilized Rotor Pooling.
    """
    def __init__(self, embed_dim, n_heads, n_layers, n_classes, use_rotor_pool=True):
        super().__init__()
        self.use_rotor_pool = use_rotor_pool
        self.blocks = nn.ModuleList([
            GeometricBlock(embed_dim, n_heads) for _ in range(n_layers)
        ])
        
        if self.use_rotor_pool:
            self.pooler = RecursiveRotorAccumulator(embed_dim)
        
        # Final classifier maps the multivector state to class logits
        self.classifier = nn.Linear(embed_dim * 32, n_classes)
        
    def forward(self, x):
        # Process through geometric layers
        for block in self.blocks:
            x = block(x)
            
        if self.use_rotor_pool:
            # Recursive Rotor Accumulation (Global State Persistence)
            x = self.pooler(x) 
        else:
            # Standard Global Average Pooling
            x = x.mean(dim=1)
        
        # Flatten multivector lanes for the linear classifier
        x = x.view(x.shape[0], -1)
        return self.classifier(x)
