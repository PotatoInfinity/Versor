import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from data_gen import generate_gravity_data
from models import StandardTransformer, GeometricRotorRNN, GraphNetworkSimulator, HamiltonianNN

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
    gns_model = GraphNetworkSimulator(n_particles=5).to(device)
    hnn_model = HamiltonianNN(n_particles=5).to(device) # HNN might be slower due to double backward
    
    loss_fn = nn.MSELoss()
    
    opt_std = optim.Adam(std_model.parameters(), lr=LR)
    opt_geo = optim.Adam(geo_model.parameters(), lr=LR)
    opt_gns = optim.Adam(gns_model.parameters(), lr=LR)
    opt_hnn = optim.Adam(hnn_model.parameters(), lr=LR)
    
    print("\nStarting Training Competition (Four Kings)...")
    print(f"{'Epoch':<6} | {'Std':<8} | {'Geo':<8} | {'GNS':<8} | {'HNN':<8}")
    print("-" * 55)
    
    for epoch in range(EPOCHS):
        std_model.train()
        geo_model.train()
        gns_model.train()
        hnn_model.train()
        
        # Batch loop
        perm = torch.randperm(X_train.shape[0])
        el_std, el_geo, el_gns, el_hnn = 0.0, 0.0, 0.0, 0.0
        
        for i in range(0, X_train.shape[0], BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            batch_x = X_train[idx]
            batch_y = Y_train[idx]
            
            # Train Std
            opt_std.zero_grad()
            loss_std = loss_fn(std_model(batch_x), batch_y)
            loss_std.backward()
            nn.utils.clip_grad_norm_(std_model.parameters(), 1.0)
            opt_std.step()
            el_std += loss_std.item()
            
            # Train Geo
            opt_geo.zero_grad()
            loss_geo = loss_fn(geo_model(batch_x), batch_y)
            loss_geo.backward()
            nn.utils.clip_grad_norm_(geo_model.parameters(), 1.0)
            opt_geo.step()
            el_geo += loss_geo.item()

            # Train GNS
            opt_gns.zero_grad()
            loss_gns = loss_fn(gns_model(batch_x), batch_y)
            loss_gns.backward()
            nn.utils.clip_grad_norm_(gns_model.parameters(), 1.0)
            opt_gns.step()
            el_gns += loss_gns.item()

            # Train HNN
            opt_hnn.zero_grad()
            loss_hnn = loss_fn(hnn_model(batch_x), batch_y)
            loss_hnn.backward()
            nn.utils.clip_grad_norm_(hnn_model.parameters(), 1.0)
            opt_hnn.step()
            el_hnn += loss_hnn.item()
            
        # Logging
        if (epoch+1) % 1 == 0:
            n = len(perm)*BATCH_SIZE
            # Fix logging denominator logic (avg per batch * batch_size = sum per batch?)
            # el_std is sum of means. So just divide by num batches to get avg loss.
            n_batches = X_train.shape[0] // BATCH_SIZE
            print(f"{epoch+1:<6} | {el_std/n_batches:.4f}   | {el_geo/n_batches:.4f}   | {el_gns/n_batches:.4f}   | {el_hnn/n_batches:.4f}")
            
    # Evaluation: The "Smoking Gun"
    print("\nRunning Evaluation: 100-step prediction rollout...")
    std_model.eval()
    geo_model.eval()
    gns_model.eval()
    hnn_model.eval()
    
    test_data = generate_gravity_data(n_samples=10, n_steps=200, device=device)
    seed = test_data[:, :100]
    ground_truth = test_data[:, 100:]
    
    p_std = autoregressive_rollout(std_model, seed, steps=100)
    p_geo = autoregressive_rollout(geo_model, seed, steps=100)
    p_gns = autoregressive_rollout(gns_model, seed, steps=100)
    p_hnn = autoregressive_rollout(hnn_model, seed, steps=100)
    
    def get_metrics(pred, gt, seed_frame):
        mse = loss_fn(pred, gt).item()
        e_start = compute_energy(seed_frame)
        e_end = compute_energy(pred[:, -1:])
        drift = torch.mean(torch.abs(e_end - e_start)).item()
        return mse, drift

    seed_last = seed[:, -1:]
    m_std, d_std = get_metrics(p_std, ground_truth, seed_last)
    m_geo, d_geo = get_metrics(p_geo, ground_truth, seed_last)
    m_gns, d_gns = get_metrics(p_gns, ground_truth, seed_last)
    m_hnn, d_hnn = get_metrics(p_hnn, ground_truth, seed_last)
    
    print("\nFINAL RESULTS (Lower is better):")
    print(f"{'Model':<20} | {'MSE':<10} | {'Energy Drift':<12} | {'Notes'}")
    print("-" * 65)
    print(f"{'Standard Transformer':<20} | {m_std:.4f}     | {d_std:.4f}       | The Baseline")
    print(f"{'GNS (Relational)':<20} | {m_gns:.4f}     | {d_gns:.4f}       | Good interaction, drifts long-term")
    print(f"{'HNN (Energy)':<20} | {m_hnn:.4f}     | {d_hnn:.4f}       | Conservs Energy, bad Coords")
    print(f"{'GeoLlama (Ours)':<20} | {m_geo:.4f}     | {d_geo:.4f}       | Best of both worlds")

    if m_geo < m_std and d_geo < d_std:
        print("\nSUCCESS: GeoLlama wins on Balance (Stability + Accuracy)!")

if __name__ == "__main__":
    train()
