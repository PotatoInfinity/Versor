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
        x_flat = x.reshape(B, S, -1) # (B, S, N*6)
        
        emb = self.embedding(x_flat) + self.pos_encoder[:, :S, :]
        
        # Create Causal Mask
        # mask[i, j] = -inf if j > i else 0
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device)
        
        out = self.transformer(emb, mask=mask)
        pred = self.head(out)
        
        return pred.reshape(B, S, N, D)

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
            
        return torch.stack(outputs, dim=1).reshape(B, S, N, D)
class GraphNetworkSimulator(nn.Module):
    """
    The Relational King.
    Treats particles as nodes, interactions as edges.
    """
    def __init__(self, n_particles=5, input_dim=6, hidden_dim=64):
        super().__init__()
        self.n_particles = n_particles
        # Node Encoder: Encodes (state) -> hidden
        self.node_enc = nn.Linear(input_dim, hidden_dim)
        
        # Edge Encoder: Encodes (rel_pos, rel_vel) -> hidden
        self.edge_enc = nn.Linear(input_dim, hidden_dim)
        
        # Message Passing (Interaction Network)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), # (Node_i, Node_j, Edge_ij)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # (Node_i, Agg_Message)
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim) # Predict acceleration/delta
        )
        
    def forward(self, x):
        # x: (B, S, N, 6)
        B, S, N, D = x.shape
        x_flat = x.reshape(B*S, N, D)
        
        # Node features
        nodes = self.node_enc(x_flat) # (BS, N, H)
        
        # Create full graph edges (N*N)
        # We can verify scaling later, for N=5 full graph is fine.
        # Edge features: x_j - x_i
        x_i = x_flat.unsqueeze(2).expand(-1, -1, N, -1)
        x_j = x_flat.unsqueeze(1).expand(-1, N, -1, -1)
        rel_x = x_j - x_i # (BS, N, N, 6)
        
        edges = self.edge_enc(rel_x) # (BS, N, N, H)
        
        # Message Passing
        # Concat (Node_i, Node_j, Edge_ij)
        n_i = nodes.unsqueeze(2).expand(-1, -1, N, -1) # (BS, N, N, H)
        n_j = nodes.unsqueeze(1).expand(-1, N, -1, -1) # (BS, N, N, H)
        
        edge_input = torch.cat([n_i, n_j, edges], dim=-1)
        messages = self.edge_mlp(edge_input) # (BS, N, N, H)
        
        # Aggregate (Sum over j)
        aggr_messages = messages.sum(dim=2) # (BS, N, H)
        
        # Update Nodes
        node_input = torch.cat([nodes, aggr_messages], dim=-1)
        delta = self.node_mlp(node_input) # (BS, N, 6)
        
        # Residual update
        next_state = x_flat + delta
        
        return next_state.reshape(B, S, N, D)

class HamiltonianNN(nn.Module):
    """
    The Energy King.
    Learns scalar Energy H(q, p).
    Equations of motion: dq/dt = dH/dp, dp/dt = -dH/dq.
    """
    def __init__(self, n_particles=5, input_dim=6, hidden_dim=128):
        super().__init__()
        self.n_particles = n_particles
        # Input is entire system state (N * 6)
        self.state_dim = n_particles * 6
        
        self.h_net = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1) # Scalar Energy
        )
        
    def forward(self, x, dt=0.01):
        # x: (B, S, N, 6)
        # HNN typically works on instantaneous state. 
        # We process each time step independently (or vectorised).
        B, S, N, D = x.shape
        x_flat = x.reshape(B * S, N * D)
        
        # Enforce gradient enabling even if in no_grad mode (for rollout)
        with torch.enable_grad():
            # Enable grad for input to compute dH/dx
            x_flat = x_flat.detach().requires_grad_(True)
            
            # Predict Energy
            energy = self.h_net(x_flat)
            
            # Compute Gradients
            grads = torch.autograd.grad(energy, x_flat, grad_outputs=torch.ones_like(energy), create_graph=True)[0]
            # grads: (BS, N*D) -> [dq1, dp1, dq2, dp2 ...]
        
        # Split into q (pos) and p (vel/momentum)
        # Assuming input is [px, py, pz, vx, vy, vz] per particle
        grads = grads.reshape(B*S, N, 6)
        dH_dq = grads[..., :3]
        dH_dp = grads[..., 3:]
        
        # Symplectic Gradients
        # dot_q = dH/dp
        # dot_p = -dH/dq
        
        dot_q = dH_dp
        dot_p = -dH_dq
        
        time_derivs = torch.cat([dot_q, dot_p], dim=-1) # (BS, N, 6)
        
        # Euler Integration Step (Predict Next)
        next_state = x_flat.reshape(B*S, N, 6) + time_derivs * dt
        
        return next_state.reshape(B, S, N, D)
