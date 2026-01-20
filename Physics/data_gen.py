import torch
import numpy as np

def generate_gravity_data(n_samples=1000, n_steps=200, n_particles=5, dt=0.01, device='cpu'):
    """
    Generates N-body gravity interaction data.
    Returns: Tensor of shape (n_samples, n_steps, n_particles, 6)
             where 6 = (px, py, pz, vx, vy, vz)
    """
    # Position and velocity initialization
    # Coordinate range: [-1, 1], Velocity range: [-0.5, 0.5]
    pos = torch.randn(n_samples, n_particles, 3, device=device) * 0.5
    vel = torch.randn(n_samples, n_particles, 3, device=device) * 0.5
    
    # Mass distribution (randomly assigned between [0.5, 1.5])
    mass = torch.rand(n_samples, n_particles, 1, device=device) + 0.5
    
    G = 1.0  # Gravitational constant
    
    trajectory = []
    
    for _ in range(n_steps):
        trajectory.append(torch.cat([pos, vel], dim=-1))
        
        # Force calculation
        # displacement: (B, N, N, 3) matrix representing r_j - r_i
        # Gravity law: F_i = \sum_j (G * m_i * m_j * (r_j - r_i) / |r_i - r_j|^3)
        
        # Shape: (B, N, 1, 3) - (B, 1, N, 3) -> (B, N, N, 3)
        diff = pos.unsqueeze(2) - pos.unsqueeze(1) 
        dist = torch.norm(diff, dim=-1, keepdim=True) + 1e-3 # Softening
        
        # F_ij = G * m_i * m_j * diff / dist^3
        # Acceleration integration: F_on_i = \sum_j G * m_i * m_j * (r_j - r_i) / |r|^3
        direction = -diff
        force_magnitude = (G * mass.unsqueeze(2) * mass.unsqueeze(1)) / (dist ** 3)
        
        # Exclusion of self-interaction effects
        mask = ~torch.eye(n_particles, device=device).bool().unsqueeze(0).unsqueeze(-1)
        force = (direction * force_magnitude * mask).sum(dim=2)
        
        # Inverse dynamics: Acceleration = Force / Mass
        acc = force / mass
        
        # Integration via Semi-implicit Euler scheme
        vel = vel + acc * dt
        
        # Boundary enforcement: Periodic boxes or reflective walls
        box_bound = 2.0
        # Wall collision detection and response
        hit_wall = (pos.abs() > box_bound)
        vel[hit_wall] *= -1.0 # Elastic collision
        pos[hit_wall] = torch.sign(pos[hit_wall]) * box_bound # Boundary clamping
        
        pos = pos + vel * dt
        
    return torch.stack(trajectory, dim=1)

if __name__ == "__main__":
    print("Generating sample data...")
    t0 = torch.tensor(0.0)
    data = generate_gravity_data(n_samples=10, n_steps=200, n_particles=3)
    print(f"Data shape: {data.shape}")
    print("Sample 0, Particle 0, Step 0-5:\n", data[0, :5, 0, :])
