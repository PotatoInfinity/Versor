import torch
import torch.nn as nn
import sys
import os

# Append current directory to system path for submodule accessibility
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import algebra
except ImportError:
    # Contingency for alternative execution environments
    try:
        from . import algebra
    except ImportError:
        import kernel as algebra

class StandardTransformer(nn.Module):
    def __init__(self, input_dim=6, n_particles=5, d_model=128, n_head=4, n_layers=2):
        super().__init__()
        self.input_dim = input_dim * n_particles
        self.embedding = nn.Linear(self.input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, d_model) * 0.1) # Stochastic learnable positional encoding
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.head = nn.Linear(d_model, self.input_dim)
        
    def forward(self, x):
        # x: (B, S, N, 6)
        B, S, N, D = x.shape
        x_flat = x.reshape(B, S, -1) # (B, S, N*6)
        
        emb = self.embedding(x_flat) + self.pos_encoder[:, :S, :]
        
        # Generation of a causal temporal mask
        # M(i, j) = -\infty \text{ if } j > i \text{ else } 0
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device)
        
        out = self.transformer(emb, mask=mask)
        pred = self.head(out)
        
        return pred.reshape(B, S, N, D)

class GeometricRotorRNN(nn.Module):
    """
    Recurrent Neural Network using Geometric Algebra.
    Optimized for Physics stability:
    1. Residual Updates (Differential state evolution).
    2. Manifold Normalization (Energy conservation constraint).
    3. Geometric Transformation Layer.
    """
    def __init__(self, input_dim=6, n_particles=5, d_mv=32, hidden_channels=16):
        super().__init__()
        self.n_particles = n_particles
        self.hidden_channels = hidden_channels
        self.d_mv = d_mv
        
        # Linear projection to the Geometric Algebra multivector state
        self.proj_in = nn.Linear(n_particles * 6, hidden_channels * 32)
        
        # State transition parameters (Multivector evolution)
        # h_{t+1} = h_t + \text{GP}(h_t, W_h) + \text{GP}(x, W_x)
        # Initialization near zero to preserve state continuity
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
        
        # Hidden state initialization
        h = torch.zeros(B, self.hidden_channels, 32, device=x.device)
        # Zero-centered initialization for residual learning stability
        
        outputs = []
        
        for t in range(S):
            x_t = x_flat[:, t, :] # (B, N*6)
            
            # Project input
            x_emb = self.proj_in(x_t).view(B, self.hidden_channels, 32)
            
            # Geometric accumulation representing force integration
            
            # Current State contribution
            rec_term = algebra.geometric_linear_layer(h, self.w_h)
            
            # Input contribution
            in_term = algebra.geometric_linear_layer(x_emb, self.w_x)
            
            # Update
            # Geometric non-linear transformation
            delta = rec_term + in_term
            
            # Application of manifold projection to ensure numerical stability
            
            # State update with manifold normalization
            h_new = h + delta
            h_new = algebra.manifold_normalization(h_new)
            
            h = h_new
            
            # State prediction via multivector projection
            out_emb = h.reshape(B, -1)
            pred_delta = self.proj_out(out_emb) 
            
            # Residual prediction facilitating coordinate continuity
            outputs.append(x_t + pred_delta)
            
        return torch.stack(outputs, dim=1).reshape(B, S, N, D)
class GraphNetworkSimulator(nn.Module):
    """
    Relational inductive bias implementation.
    Models particles as nodes and interactions as graph edges.
    """
    def __init__(self, n_particles=5, input_dim=6, hidden_dim=64):
        super().__init__()
        self.n_particles = n_particles
        # Node Encoder: Encodes (state) -> hidden
        self.node_enc = nn.Linear(input_dim, hidden_dim)
        
        # Interaction encoding from relative state vectors
        self.edge_enc = nn.Linear(input_dim, hidden_dim)
        
        # Message Passing (Interaction Network)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), # (Node_i, Node_j, Edge_ij)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # (Node state, aggregated messages)
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim) # Prediction of coordinate delta
        )
        
    def forward(self, x):
        # x: (B, S, N, 6)
        B, S, N, D = x.shape
        x_flat = x.reshape(B*S, N, D)
        
        # Node features
        nodes = self.node_enc(x_flat) # (BS, N, H)
        
        # Fully connected graph construction
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
    Hamiltonian Neural Network implementation.
    Parametrizes the system Hamiltonian H(q, p) to derive symplectic motion.
    Fundamental equations: \dot{q} = \partial H / \partial p, \dot{p} = -\partial H / \partial q.
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
        # Computation of instantaneous system energy across temporal states
        B, S, N, D = x.shape
        x_flat = x.reshape(B * S, N * D)
        
        # Gradient computation required for Hamiltonian mechanics
        with torch.enable_grad():
            # Enable grad for input to compute dH/dx
            x_flat = x_flat.detach().requires_grad_(True)
            
            # Predict Energy
            energy = self.h_net(x_flat)
            
            # Gradient extraction for symplectic integration
        
        # Decomposition into canonical coordinates (pos) and momenta (vel)
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
