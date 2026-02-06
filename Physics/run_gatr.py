
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import sys
import os
import json

# Add paths
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root_dir))
sys.path.append(os.path.join(root_dir, "gatr"))
sys.path.append(os.path.join(root_dir, "Physics"))

from data_gen import generate_gravity_data
import mock_dependencies
mock_dependencies.apply_mocks()

# Import GATr components
try:
    from gatr import GATr, SelfAttentionConfig, MLPConfig
    from gatr.interface import embed_point, embed_scalar, embed_translation, extract_point, extract_translation
    from gatr.utils.einsum import enable_cached_einsum
    
    # Disable opt_einsum dependency
    enable_cached_einsum(False)

except ImportError as e:
    print(f"Failed to import GATr: {e}")
    print("Ensure 'gatr' folder is in the root directory.")
    sys.exit(1)

def compute_energy(data, mass=1.0, G=1.0):
    """
    Computes total energy of the system.
    Data: (B, T, N, 6) -> (pos, vel)
    Returns: (B, T) energy
    """
    pos = data[..., :3]
    vel = data[..., 3:]
    
    v_sq = torch.sum(vel**2, dim=-1) # (B, T, N)
    ke = 0.5 * torch.sum(v_sq, dim=-1) # (B, T)
    
    pe = torch.zeros_like(ke)
    B, T, N, _ = pos.shape
    
    for i in range(N):
        for j in range(i + 1, N):
            diff = pos[..., i, :] - pos[..., j, :]
            dist = torch.norm(diff, dim=-1) + 1e-3
            pe -= (G * 1.0 * 1.0) / dist
            
    return ke + pe

class GATrAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        # GATr configuration for N-Body
        self.gatr = GATr(
            in_mv_channels=1, 
            out_mv_channels=1,
            in_s_channels=None,
            out_s_channels=None,
            hidden_mv_channels=16,
            hidden_s_channels=32,
            num_blocks=10, 
            attention=SelfAttentionConfig(), 
            mlp=MLPConfig()
        )
        
    def forward(self, x):
        # x: (B, S, N, 6)
        B, S, N, D = x.shape
        x_flat = x.reshape(B*S, N, D)
        
        # Add mass (dummy 1.0)
        mass = torch.ones(B*S, N, 1, device=x.device)
        inputs = torch.cat([mass, x_flat], dim=-1) # (BS, N, 7)
        
        # Embed
        masses = embed_scalar(inputs[..., 0:1])
        points = embed_point(inputs[..., 1:4])
        velocities = embed_translation(inputs[..., 4:7])
        
        # Sum embeddings (PGA convention in GATr)
        mv_in = (masses + points + velocities).unsqueeze(2) # (BS, N, 1, 16)
        
        # Forward
        mv_out, _ = self.gatr(mv_in, scalars=None) 
        
        # Extract
        pred_pos = extract_point(mv_out[:, :, 0, :])
        pred_vel = extract_translation(mv_out[:, :, 0, :])
        
        next_state = torch.cat([pred_pos, pred_vel], dim=-1)
        
        return next_state.reshape(B, S, N, D)

def run_gatr_experiment():
    seeds = [42, 123, 456]
    mses = []
    drifts = []
    latencies_all = []
    
    print("="*60)
    print("RUNNING GATr (Table 2) - Multi-Seed")
    print("="*60)

    for seed in seeds:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nSeed: {seed}")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Hyperparams
        BATCH_SIZE = 8
        STEPS = 50 
        EPOCHS = 5
        LR = 5e-4 
        
        train_data = generate_gravity_data(n_samples=100, n_steps=STEPS, device=device)
        X_train = train_data[:, :-1]
        Y_train = train_data[:, 1:]
        
        model = GATrAdapter().to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.MSELoss()
        
        # Train
        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(X_train.shape[0])
            for i in range(0, X_train.shape[0], BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                batch_x = X_train[idx]
                batch_y = Y_train[idx]
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        
        # Evaluate
        model.eval()
        test_data = generate_gravity_data(n_samples=10, n_steps=100, device=device)
        seed_window = test_data[:, :50]
        ground_truth = test_data[:, 50:]
        
        # Latency measurement
        latencies = []
        with torch.no_grad():
            for _ in range(50):
                start = time.time()
                _ = model(seed_window[:1])
                latencies.append((time.time() - start) * 1000) # ms
        
        avg_latency = np.mean(latencies)
        
        current_seq = seed_window
        preds = []
        with torch.no_grad():
            curr = current_seq[:, -1:]
            for _ in range(50):
                next_step = model(curr)
                preds.append(next_step)
                curr = next_step
                
        preds = torch.cat(preds, dim=1)
        mse = loss_fn(preds, ground_truth).item()
        
        # Energy Drift
        seed_last = seed_window[:, -1:]
        e_start = compute_energy(seed_last)
        e_end = compute_energy(preds[:, -1:])
        drift_pct = torch.mean(torch.abs(e_end - e_start) / (torch.abs(e_start) + 1e-6)).item() * 100
        
        print(f"Seed {seed} -> MSE: {mse:.4f}, Drift: {drift_pct:.2f}%, Latency: {avg_latency:.2f}ms")
        mses.append(mse)
        drifts.append(drift_pct)
        latencies_all.append(avg_latency)
        
    print("\nGATr RESULTS:")
    print(f"MSE: {np.mean(mses):.2f} ± {np.std(mses):.2f}")
    print(f"Drift: {np.mean(drifts):.2f} ± {np.std(drifts):.2f}%")
    print(f"Latency: {np.mean(latencies_all):.2f} ms")
    
    with open("Physics/results/gatr_stats.json", "w") as f:
        json.dump({
            "mean_mse": np.mean(mses),
            "std_mse": np.std(mses),
            "mean_drift": np.mean(drifts),
            "std_drift": np.std(drifts),
            "mean_latency": np.mean(latencies_all)
        }, f)

if __name__ == "__main__":
    run_gatr_experiment()
