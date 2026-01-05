import torch
import torch.nn as nn
from .cga import (
    batch_geometric_product, exp_map, inverse, normalize_rotor, 
    VECTOR_INDICES, BIVECTOR_INDICES, QUADVECTOR_INDICES
)

class ManifoldMixingLayer(nn.Module):
    """
    Implements 'Manifold Mixing' (Section 9.2).
    """
    def __init__(self, num_heads=64):
        super().__init__()
        self.num_heads = num_heads
        self.mixer = nn.Linear(num_heads, num_heads, bias=False)
        
        with torch.no_grad():
            self.mixer.weight.copy_(torch.eye(num_heads))
            self.mixer.weight.add_(torch.randn(num_heads, num_heads) * 0.01)

    def forward(self, psi):
        # psi: (Batch, Heads, 32)
        # Transpose to mix heads: (Batch, 32, Heads)
        psi_t = psi.transpose(1, 2) 
        mixed_t = self.mixer(psi_t)
        return mixed_t.transpose(1, 2) # Back to (Batch, Heads, 32)

class GeoLlamaState(nn.Module):
    """
    Maintains the structural context PSI (Context Rotor) for the Geo-Llama stream.
    Now supports Batched Backpropagation (TBPTT).
    """
    def __init__(self, num_heads=64, d_model=2048, device='cpu'):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.device = device
        # PSI is not a Parameter, it's a buffer (state)
        # Shape: (Batch, Heads, 32) - Initialized to None until reset
        self.psi = None 

    def reset_state(self, batch_size=1):
        """Initializes PSI to Identity (Scalar=1) for a new batch."""
        self.psi = torch.zeros((batch_size, self.num_heads, 32), device=self.device)
        self.psi[:, :, 0] = 1.0 # Scalar part = 1
        return self.psi

    def update(self, rotors, mixing_layer=None):
        """
        Recursive Rotor Accumulation: PSI_t = R_t * PSI_t-1 * R_t_inv
        rotors: (Batch, Heads, 32)
        """
        if self.psi is None:
            # Auto-init if not reset (assumes batch size from rotors)
            self.reset_state(batch_size=rotors.shape[0])

        # 1. Inverse of the incoming rotors
        rotors_inv = inverse(rotors)
        
        # 2. Update PSI: PSI = R * PSI * R_inv
        # batch_geometric_product handles (Batch, Heads, 32)
        temp = batch_geometric_product(rotors, self.psi)
        new_psi = batch_geometric_product(temp, rotors_inv)
        
        # 3. Manifold Correction (Drift Mitigation)
        # Crucial: We normalize BEFORE saving state to keep gradients stable
        new_psi = normalize_rotor(new_psi)
        
        # 4. Manifold Mixing
        if mixing_layer is not None:
            new_psi = mixing_layer(new_psi)
            new_psi = normalize_rotor(new_psi)
        
        # 5. Save state (NO DETACH - preserves gradient graph)
        self.psi = new_psi
        return self.psi

class SpecializedLiftingLayer(nn.Module):
    """
    Implements 'Head Specialization' (Section 2.1).
    Partitions 64 heads into different functional manifolds:
    - Heads 0-9: Syntactic (10 Bivectors, Small Scale)
    - Heads 10-39: Semantic (5 Vectors + 10 Bivectors, Medium Scale)
    - Heads 40-63: Narrative (10 Bivectors + 5 Quadvectors, Large Scale)
    """
    def __init__(self, d_model=2048, num_heads=64):
        super().__init__()
        assert num_heads == 64, "Specialization logic tuned for 64 heads"
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Dimensions per group
        self.syntax_heads = slice(0, 10)
        self.semantic_heads = slice(10, 40)
        self.narrative_heads = slice(40, 64)
        
        # Projections
        # Syntax: 10 bivectors
        self.proj_syntax = nn.Linear(d_model, 10 * 10)
        # Semantic: 5 vectors + 10 bivectors = 15
        self.proj_semantic = nn.Linear(d_model, 30 * 15)
        # Narrative: 10 bivectors + 5 quadvectors = 15
        self.proj_narrative = nn.Linear(d_model, 24 * 15)
        
        # Initialization Scales (Section 2.1: short, medium, long scale)
        with torch.no_grad():
            self.proj_syntax.weight *= 0.1  # Short-scale
            self.proj_semantic.weight *= 0.5 # Medium-scale
            self.proj_narrative.weight *= 1.5 # Long-scale (global context)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        returns: (batch, seq_len, 64, 32) - Rotors
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype
        
        # Project each group
        syn_out = self.proj_syntax(x).view(batch_size, seq_len, 10, 10)
        sem_out = self.proj_semantic(x).view(batch_size, seq_len, 30, 15)
        nar_out = self.proj_narrative(x).view(batch_size, seq_len, 24, 15)
        
        # Map to full 32-lane multivectors
        B_full = torch.zeros(batch_size, seq_len, 64, 32, device=device, dtype=dtype)
        
        # Syntax mapping
        for i, idx in enumerate(BIVECTOR_INDICES):
            B_full[..., 0:10, idx] = syn_out[..., i]
            
        # Semantic mapping
        for i, idx in enumerate(VECTOR_INDICES):
            B_full[..., 10:40, idx] = sem_out[..., i]
        for i, idx in enumerate(BIVECTOR_INDICES):
            B_full[..., 10:40, idx] = sem_out[..., i + 5]
            
        # Narrative mapping
        for i, idx in enumerate(BIVECTOR_INDICES):
            B_full[..., 40:64, idx] = nar_out[..., i]
        for i, idx in enumerate(QUADVECTOR_INDICES):
            B_full[..., 40:64, idx] = nar_out[..., i + 10]
            
        # Map to Rotors (Using updated exp_map from cga.py)
        # Apply tanh to clamp magnitudes before exp_map for stability
        B_full = torch.tanh(B_full) * 3.14159  # Clamp to +/- PI
        
        Rotors = exp_map(-B_full / 2.0)
        return Rotors
    
    def lift(self, x):
        return self.forward(x)

class GeometricLiftingLayer(SpecializedLiftingLayer):
    """Alias for backwards compatibility if needed, now specialized."""
    pass

def geometry_conditioned_attention_bias(psi, Q_lifted, K_lifted, lambda_val=0.1):
    # (Keep your existing function, it works fine for logic)
    from .cga import batch_wedge_product, batch_inner_product
    rel_plane = batch_wedge_product(Q_lifted.unsqueeze(3), K_lifted.unsqueeze(2))
    
    # Broadcast psi: (Batch, Heads, 1, 1, 32)
    if psi.dim() == 3: # (Batch, Heads, 32)
        psi_exp = psi.view(psi.shape[0], psi.shape[1], 1, 1, 32)
    elif psi.dim() == 2: # Old (Heads, 32) fallback
        psi_exp = psi.view(1, psi.shape[0], 1, 1, 32)
        
    bias = batch_inner_product(psi_exp, rel_plane)
    return lambda_val * bias
