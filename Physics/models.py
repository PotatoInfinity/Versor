import torch
import torch.nn as nn
import sys
import os

# Add current directory to path to import algebra
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import algebra
except ImportError:
    # Fallback if running from a different root
    try:
        from . import algebra
    except ImportError:
        import kernel as algebra

class StandardTransformer(nn.Module):
    def __init__(self, input_dim=6, n_particles=5, d_model=128, n_head=4, n_layers=2):
        super().__init__()
        self.input_dim = input_dim * n_particles
        self.embedding = nn.Linear(self.input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, d_model) * 0.1) # Simple learnable pos encoding
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.head = nn.Linear(d_model, self.input_dim)
        
    def forward(self, x):
        # x: (B, S, N, 6)
        B, S, N, D = x.shape
        x_flat = x.view(B, S, -1) # (B, S, N*6)
        
        emb = self.embedding(x_flat) + self.pos_encoder[:, :S, :]
        
        # Create Causal Mask
        # mask[i, j] = -inf if j > i else 0
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device)
        
        out = self.transformer(emb, mask=mask)
        pred = self.head(out)
        
        return pred.view(B, S, N, D)

class GeometricRotorRNN(nn.Module):
    """
    Recurrent Neural Network using Geometric Algebra.
    Optimized for Physics stability:
    1. Residual Updates (Newton's Law: state changes slightly).
    2. Manifold Normalization (Keeps energies bounded).
    3. Geometric Gating (Element-wise sigmoid on the MV magnitude? Or just linear).
    """
    def __init__(self, input_dim=6, n_particles=5, d_mv=32, hidden_channels=16):
        super().__init__()
        self.n_particles = n_particles
        self.hidden_channels = hidden_channels
        self.d_mv = d_mv
        
        # Input: Flatten all particles -> Vector Embeddings
        # We assume input is a vector (Grade 1). 
        # We define a learnable projection to internal MV state.
        self.proj_in = nn.Linear(n_particles * 6, hidden_channels * 32)
        
        # State Update weights (Rotor evolution)
        # h_new = h_old + GeometricProduct(h_old, W_h) + GeometricProduct(x, W_x)
        # We initialize W_h to be small => state persistence.
        self.w_h = nn.Parameter(torch.randn(hidden_channels, hidden_channels, 32) * 0.01)
        self.w_x = nn.Parameter(torch.randn(hidden_channels, hidden_channels, 32) * 0.01)
        
        # Output projection
        self.proj_out = nn.Linear(hidden_channels * 32, n_particles * 6)
        
        # Initialize biases for projections
        self.proj_in.bias.data.fill_(0)
        self.proj_out.bias.data.fill_(0)
        
    def forward(self, x):
        # x: (B, S, N, 6)
        B, S, N, D = x.shape
        x_flat = x.view(B, S, -1)
        
        # Initial Hidden State (Identity/Vacuum)
        h = torch.zeros(B, self.hidden_channels, 32, device=x.device)
        # Set scalar part to 1 for identity-like behavior if treating as rotor?
        # Let's keep it zero-centered for residual learning.
        
        outputs = []
        
        for t in range(S):
            x_t = x_flat[:, t, :] # (B, N*6)
            
            # Project input
            x_emb = self.proj_in(x_t).view(B, self.hidden_channels, 32)
            
            # Geometric Residual Update
            # Instead of replacing H, we accumulate into it.
            # This models integration ( Sum of forces ).
            
            # Current State contribution
            rec_term = algebra.geometric_linear_layer(h, self.w_h)
            
            # Input contribution
            in_term = algebra.geometric_linear_layer(x_emb, self.w_x)
            
            # Update
            # h_new = h + tanh(updates) ?
            # Standard Transformers succeed because of Residuals + LayerNorm.
            # We use Residuals + ManifoldNorm.
            
            delta = rec_term + in_term
            
            # Apply geometric non-linearity? 
            # The geometric product itself is the non-linearity if we had higher order terms.
            # But here we just did Linear * Linear.
            # Let's add a "Sandwich" or "Manifold Norm" non-linearity.
            
            # Stable Update:
            h_new = h + delta
            
            # Normalization (The "Constraint" enforcement)
            # Keeps the hidden state on the manifold (prevents explosion).
            h_new = algebra.manifold_normalization(h_new)
            
            h = h_new
            
            # Prediction
            # Predict DELTA state (Residual Physics)
            # x_{t+1} = x_t + predicted_delta
            # But the model wrapper expects full state.
            # Let's make the RNN predict the FULL state for compatibility with loss,
            # but rely on the RNN state `h` being the integrator.
            
            out_emb = h.reshape(B, -1)
            pred_delta = self.proj_out(out_emb) 
            
            # If we want the network to be purely residual on the OUTPUT:
            # pred_next = x_t + pred_delta
            # This is a huge bias towards "Physics is continuous".
            # Standard Transformer has to learn this. We force it.
            
            outputs.append(x_t + pred_delta)
            
        return torch.stack(outputs, dim=1).view(B, S, N, D)