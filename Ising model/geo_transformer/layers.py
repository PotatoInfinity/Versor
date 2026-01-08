import torch
import torch.nn as nn
from .core import gp_cl41, wedge_cl41, inner_cl41, normalize_cl41, GRADE_INDICES, get_gp_map

class GeometricLinear(nn.Module):
    """
    Multivector Linear Layer for Clifford Algebra Cl(4,1).
    
    Weights are multivectors (32-lane). The operation is a contraction 
    using the Geometric Product (GP). This maintains geometric covariance 
    throughout the linear transformation.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Initialization using grade-aware variance scaling
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, 32))
        
        self.reset_parameters()

    def reset_parameters(self):
        """
        Grade-Aware Xavier Initialization.
        Initializes scalar and vector components to maintain signal variance.
        """
        with torch.no_grad():
            std = 1.0 / (self.in_features * 32)**0.5
            # Initialize Scalar (Grade 0)
            self.weight.data[:, :, 0].normal_(0.0, std)
            # Initialize Vectors (Grade 1): e1, e2, e3, e+, e-
            for idx in GRADE_INDICES[1]:
                self.weight.data[:, :, idx].normal_(0.0, std)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input multivector of shape (..., in_features, 32)
        Returns:
            Tensor: Output multivector of shape (..., out_features, 32)
        """
        # Two-step contraction for better MPS/Accelerator stability
        # Step 1: Combine weight features with input multivectors
        # (out, in, 32) x (..., in, 32) -> (..., out, 32, 32)
        res = torch.einsum('o i j, ... i l -> ... o j l', self.weight, x)
        
        # Step 2: Apply the Geometric Product Cayley table
        # (..., out, 32, 32) x (32, 32, 32) -> (..., out, 32)
        gp_map = get_gp_map(x.device, x.dtype)
        out = torch.einsum('... o j l, j l k -> ... o k', res, gp_map)
        
        # Manifold projection for numerical stability
        return normalize_cl41(out)

    def __repr__(self):
        return f"GeometricLinear(in_features={self.in_features}, out_features={self.out_features})"


class GeometricAttention(nn.Module):
    """
    Geometric Product Attention (GPA).
    
    Instead of standard dot-product attention, GPA uses the full Geometric
    Product Q * K to compute attention scores. This incorporates:
    1.  The Scalar Projection (Standard Attention).
    2.  The Bivector Rotation (Geometric Coupling).
    
    This allows the model to attend to "orientational" features in GA space.
    """
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        self.q_proj = GeometricLinear(embed_dim, embed_dim)
        self.k_proj = GeometricLinear(embed_dim, embed_dim)
        self.v_proj = GeometricLinear(embed_dim, embed_dim)
        self.o_proj = GeometricLinear(embed_dim, embed_dim)
        
        # Scaling parameter for the bivector influence
        self.attn_lambda = nn.Parameter(torch.tensor(0.1))
        self.bivector_indices = GRADE_INDICES[2]
        
    def forward(self, x):
        """
        Args:
            x (Tensor): Multivector sequence (B, S, D, 32)
        """
        batch, seq, embed_dim, _ = x.shape
        
        # Project and restructure for Multi-Head Attention
        q = self.q_proj(x).view(batch, seq, self.n_heads, self.head_dim, 32).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.n_heads, self.head_dim, 32).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.n_heads, self.head_dim, 32).transpose(1, 2)
        
        # Optimized Direct Contraction Scoring: Pairwise GP summed over head_dim (d)
        # We break this into two steps to avoid MPS driver hangs on complex einsums
        # Step 1: Pairwise feature product
        # (B, H, S, D, 32) x (B, H, X, D, 32) -> (B, H, S, X, 32, 32)
        pair_prod = torch.einsum('b h s d i, b h x d j -> b h s x i j', q, k)
        
        # Step 2: Contract with Cayley table
        gp_map = get_gp_map(q.device, q.dtype)
        raw_mv = torch.einsum('b h s x i j, i j k -> b h s x k', pair_prod, gp_map)
        raw_mv = normalize_cl41(raw_mv)
        
        # Score = <Q*K>_0 + lambda * ||<Q*K>_2||
        scalar_part = raw_mv[..., 0]
        bivector_part = raw_mv[..., self.bivector_indices]
        bivector_norm = torch.sqrt(torch.sum(bivector_part**2, dim=-1) + 1e-8)
        
        score = (scalar_part + self.attn_lambda * bivector_norm) / (self.head_dim ** 0.5)
        attn_probs = torch.softmax(score, dim=-1)
        
        # Weighted accumulation of Value multivectors
        out = torch.einsum('b h s i , b h i d l -> b h s d l', attn_probs, v)
        
        # Recombine heads and final projection
        out = out.transpose(1, 2).contiguous().view(batch, seq, embed_dim, 32)
        return self.o_proj(out)

    def __repr__(self):
        return f"GeometricAttention(embed_dim={self.embed_dim}, heads={self.n_heads})"

