import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from data_gen import generate_gravity_data
from models import StandardTransformer, GeometricRotorRNN

def compute_energy(data, mass=1.0, G=1.0):
    """
    Computes total energy of the system.
    Data: (B, T, N, 6) -> (pos, vel)
    Returns: (B, T) energy
    """
    pos = data[..., :3]
    vel = data[..., 3:]
    
    # Kinetic Energy: 0.5 * m * v^2
    # Assuming mass=1.0 for simplicity or we need to pass masses.
    # In data_gen, masses are random. For metric tracking, we can assume avg mass or just check conservation relative to t=0.
    # Let's approximate mass=1.0 for the metric if exact mass isn't available, 
    # or just track stability (exploding coordinates).
    
    v_sq = torch.sum(vel**2, dim=-1) # (B, T, N)
    ke = 0.5 * torch.sum(v_sq, dim=-1) # (B, T)
    
    # Potential Energy: - G * mi * mj / r
    pe = torch.zeros_like(ke)
    B, T, N, _ = pos.shape
    
    # Pairwise
    for i in range(N):
        for j in range(i + 1, N):
            diff = pos[..., i, :] - pos[..., j, :]
            dist = torch.norm(diff, dim=-1) + 1e-3
            pe -= (G * 1.0 * 1.0) / dist
            
    return ke + pe

def autoregressive_rollout(model, seed_data, steps=100):
    """
    Predicts next 'steps' frames using the model autoregressively.
    seed_data: (B, Seed_Steps, N, 6)
    """
    current_seq = seed_data
    preds = []
    
    with torch.no_grad():
        for _ in range(steps):
            # Predict next step based on history
            # Model forward expects full sequence, returns full sequence predictions.
            # We take the last prediction.
            
            # Optimization: If model supports cache, use it. Standard Transformer/RNN usually slow O(L^2) or O(L) redraw.
            # For this demo, full forward pass is fine for short seq.
            
            out = model(current_seq)
            next_step = out[:, -1:, :, :] # (B, 1, N, 6)
            preds.append(next_step)
            
            current_seq = torch.cat([current_seq, next_step], dim=1)
            # Optional: Windowing to keep context fixed size?
            # Model trained on 100 steps. If we feed 101, 102...
            # Standard Transformer handles variable length. RNN handles variable length.
            # But let's keep window if it gets too long, or just grow it.
            if current_seq.shape[1] > 100:
                current_seq = current_seq[:, -100:, :, :]
                
    return torch.cat(preds, dim=1)

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Mac optimization: use mps if available?
    # MLX is handled in kernel, but torch mps is separate.
    # For now keep cpu or cuda.
    
    print(f"Using device: {device}")
    
    # Hyperparams
    BATCH_SIZE = 16
    STEPS = 100
    EPOCHS = 30 # Longer run for stability
    LR = 1e-3
    
    # Generate Training Data
    print("Generating training data...")
    train_data = generate_gravity_data(n_samples=200, n_steps=STEPS, device=device)
    val_data = generate_gravity_data(n_samples=50, n_steps=STEPS, device=device)
    
    X_train = train_data[:, :-1]
    Y_train = train_data[:, 1:]
    
    # Init Models
    std_model = StandardTransformer(n_particles=5).to(device)
    geo_model = GeometricRotorRNN(n_particles=5).to(device)
    
    loss_fn = nn.MSELoss()
    
    opt_std = optim.Adam(std_model.parameters(), lr=LR)
    opt_geo = optim.Adam(geo_model.parameters(), lr=LR)
    
    print("\nStarting Training Competition...")
    print(f"{'Epoch':<6} | {'Std Loss':<10} | {'Geo Loss':<10}")
    print("-" * 35)
    
    for epoch in range(EPOCHS):
        std_model.train()
        geo_model.train()
        
        # Batch loop
        perm = torch.randperm(X_train.shape[0])
        epoch_loss_std = 0.0
        epoch_loss_geo = 0.0
        
        for i in range(0, X_train.shape[0], BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            batch_x = X_train[idx]
            batch_y = Y_train[idx]
            
            # Train Std
            opt_std.zero_grad()
            pred_std = std_model(batch_x)
            loss_std = loss_fn(pred_std, batch_y)
            loss_std.backward()
            nn.utils.clip_grad_norm_(std_model.parameters(), 1.0)
            opt_std.step()
            epoch_loss_std += loss_std.item()
            
            # Train Geo
            opt_geo.zero_grad()
            pred_geo = geo_model(batch_x)
            loss_geo = loss_fn(pred_geo, batch_y)
            loss_geo.backward()
            nn.utils.clip_grad_norm_(geo_model.parameters(), 1.0)
            opt_geo.step()
            epoch_loss_geo += loss_geo.item()
            
        # Logging
        if (epoch+1) % 1 == 0:
            print(f"{epoch+1:<6} | {epoch_loss_std/len(perm)*BATCH_SIZE:.6f}   | {epoch_loss_geo/len(perm)*BATCH_SIZE:.6f}")
            
    # Evaluation: The "Smoking Gun"
    print("\nRunning Evaluation: 100-step prediction rollout...")
    std_model.eval()
    geo_model.eval()
    
    # Take a validation sample (first 50 frames as context, predict next 50)
    # Actually prompt says: Feed first 100, predict next 100.
    test_data = generate_gravity_data(n_samples=10, n_steps=200, device=device)
    seed = test_data[:, :100]
    ground_truth = test_data[:, 100:]
    
    pred_std = autoregressive_rollout(std_model, seed, steps=100)
    pred_geo = autoregressive_rollout(geo_model, seed, steps=100)
    
    # 1. MSE
    mse_std = loss_fn(pred_std, ground_truth).item()
    mse_geo = loss_fn(pred_geo, ground_truth).item()
    
    # 2. Energy Concept
    # We check if energy explodes.
    # Compute initial energy at t=100
    e_start = compute_energy(seed[:, -1:])
    
    # Compute energy at t=200 (end of rollout)
    e_end_std = compute_energy(pred_std[:, -1:])
    e_end_geo = compute_energy(pred_geo[:, -1:])
    
    drift_std = torch.mean(torch.abs(e_end_std - e_start)).item()
    drift_geo = torch.mean(torch.abs(e_end_geo - e_start)).item()
    
    print("\nResults:")
    print(f"Standard Transformer | MSE: {mse_std:.6f} | Energy Drift: {drift_std:.6f}")
    print(f"GeoLlama Rotor RNN   | MSE: {mse_geo:.6f} | Energy Drift: {drift_geo:.6f}")
    
    if mse_geo < mse_std and drift_geo < drift_std:
        print("\nSUCCESS: GeoLlama outperforms Standard Transformer in Physics!")
    else:
        print("\nAnalysis required. Models might need tuning.")

if __name__ == "__main__":
    train()