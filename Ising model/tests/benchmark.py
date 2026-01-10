import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import time
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geo_transformer.core import conformal_lift
from geo_transformer.model import GeometricTransformer

# =================================================================
# 1. RESEARCH CONFIGURATION (Matching CUDA Benchmark)
# =================================================================
EMBED_DIM = 4   
TARGET_RATIO = 1.3
BATCH_SIZE = 16
LR = 0.0007
EPOCHS = 40
N_LAYERS = 2
N_HEADS = 2
EXPANSION = 6
N_CLASSES = 3

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- REPRODUCTION LOCKDOWN ---
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

# =================================================================
# 2. COMPARISON MODELS
# =================================================================
class VanillaTransformer(nn.Module):
    def __init__(self, d_model=96, nhead=2, n_layers=2, d_ff=None):
        super().__init__()
        if d_ff is None: d_ff = d_model * 2
        self.proj = nn.Linear(4*32, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_ff, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 3)
    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1) 
        return self.head(self.transformer(self.proj(x)).mean(dim=1))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =================================================================
# 3. TRAINING ENGINE
# =================================================================
def train_model(model, loader, val_loader, model_name):
    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    hist = {'loss': [], 'acc': []}
    
    for e in range(EPOCHS):
        model.train(); tl = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); l = crit(model(x), y); l.backward(); opt.step(); tl += l.item()
            
        model.eval(); corr, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                corr += (model(x).argmax(1) == y).sum().item(); total += y.size(0)
        
        acc = corr/total
        hist['loss'].append(tl/len(loader)); hist['acc'].append(acc)
        print(f"Epoch {e:2d} | Val Acc: {acc:.4f}")
        
        # Specific Early Exit from CUDA Benchmark
        if acc > 0.985 and e > 12: break
        
    # Capture peak memory (MB)
    peak_vram = 0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        # Estimation for Mac/MPS
        params = count_parameters(model)
        act_estimate = (BATCH_SIZE * 1024 * EMBED_DIM * 32 * 4) / (1024 * 1024)
        peak_vram = (params * 4) / (1024 * 1024) + act_estimate + 38.0
        
    return hist, peak_vram

# =================================================================
# 4. MASTER APP
# =================================================================
def run_master_benchmark(seed=None):
    if seed is None:
        seed = random.randint(0, 1000)
    print(f"Executing Research Seed: {seed}")
    
    # Set seeds (Natural Flow)
    set_all_seeds(seed)
    
    data_path = "/Users/mac/Desktop/Geo-llama/Research/data/ising_data.pt"
    if not os.path.exists(data_path):
        print("Upload 'ising_data.pt' first!"); return

    ckpt = torch.load(data_path, map_location='cpu')
    X, Y = ckpt['data'].float().to(DEVICE), ckpt['labels'].to(DEVICE)
    X_in = conformal_lift(X.view(X.shape[0],-1)).unsqueeze(2).repeat(1, 1, EMBED_DIM, 1)
    
    ds = TensorDataset(X_in, Y)
    # Natural Flow Split
    train_ds, val_ds = random_split(ds, [45, len(ds)-45]) 
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # 1. Vanilla Capacity Search
    geo = GeometricTransformer(EMBED_DIM, N_HEADS, N_LAYERS, N_CLASSES, expansion=EXPANSION).to(DEVICE)
    g_p = count_parameters(geo)
    
    target_p = g_p * TARGET_RATIO
    v_dim = 32
    for d in range(24, 128, 4):
        test_v = VanillaTransformer(d_model=d, nhead=2, n_layers=2, d_ff=d*2)
        if count_parameters(test_v) > target_p:
            v_dim = d - 4
            break
            
    van = VanillaTransformer(d_model=v_dim, nhead=2, n_layers=2, d_ff=v_dim*2).to(DEVICE)
    v_p = count_parameters(van)
    
    print(f"Geo({g_p:,}) vs Vanilla({v_p:,}) ~{v_p/g_p:.1f}x larger")

    v_h, v_vram = train_model(van, train_ld, val_ld, "Vanilla")
    g_h, g_vram = train_model(geo, train_ld, val_ld, "Geo-Llama")

    # --- DYNAMIC PHASE ACCURACY EVALUATION ---
    def get_phase_acc(model, loader):
        model.eval(); phase_res = {0: [], 1: [], 2: []}
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb).argmax(1)
                for i in range(len(yb)): phase_res[yb[i].item()].append((preds[i] == yb[i]).item())
        return [np.mean(phase_res[i])*100 for i in range(3)]

    v_ph = get_phase_acc(van, val_ld); g_ph = get_phase_acc(geo, val_ld)
    
    def get_s(acc_hist):
        pk = max(acc_hist)
        return next(i for i, v in enumerate(acc_hist) if v >= 0.99*pk)
        
    v_s = get_s(v_h['acc'])
    g_s = get_s(g_h['acc'])

    # --- PLOTTING DASHBOARD ---
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor='white')
    v_c, g_c, labels = "#d67a7b", "#7395d6", ['Standard (Heavy)', 'Geo-Llama']

    # Chart 1: Params (Matched context)
    axes[0,0].bar(labels, [v_p, g_p], color=[v_c, g_c], alpha=0.9, width=0.5)
    axes[0,0].set_title('Parameters (Capacity Match)', fontweight='bold')
    axes[0,0].set_ylim(0, max(v_p, g_p)*1.6)
    for i, v in enumerate([v_p, g_p]): axes[0,0].text(i, v + (max(v_p, g_p)*0.08), f"{v:,}", ha='center', fontweight='bold')

    # Chart 2: Grokking Speed (Epochs)
    axes[0,1].bar(labels, [v_s, g_s], color=[v_c, g_c], alpha=0.9, width=0.5)
    axes[0,1].set_title('Learning Speed (Epochs to Peak)', fontweight='bold')
    axes[0,1].set_ylim(0, max(v_s, g_s)*1.6)
    for i, v in enumerate([v_s, g_s]): axes[0,1].text(i, v + (max(v_s, g_s)*0.08), f"{v} Epochs", ha='center', fontweight='bold')

    # Chart 3: Storage Footprint (Disk KB)
    v_disk = (v_p * 4) / 1024
    g_disk = (g_p * 4) / 1024
    disk_vals = [v_disk, g_disk]
    axes[0,2].bar(labels, disk_vals, color=[v_c, g_c], alpha=0.9, width=0.5)
    axes[0,2].set_title('Storage Footprint (KB)', fontweight='bold')
    axes[0,2].set_ylim(0, max(disk_vals)*1.6)
    for i, v in enumerate(disk_vals): axes[0,2].text(i, v + (max(disk_vals)*0.08), f"{v:.1f}KB", ha='center', fontweight='bold')

    # Row 2: Convergence History
    axes[1,0].plot(v_h['acc'], color=v_c, lw=3, label='Standard', marker='o', markersize=4)
    axes[1,0].plot(g_h['acc'], color=g_c, lw=3, label='Geo-Llama', marker='s', markersize=4)
    axes[1,0].set_title('Few-Shot Accuracy (N=45)', fontweight='bold'); axes[1,0].legend(); axes[1,0].grid(alpha=0.15)

    axes[1,1].plot(v_h['loss'], color=v_c, lw=3); axes[1,1].plot(g_h['loss'], color=g_c, lw=3)
    axes[1,1].set_title('Loss Decay Stabilization', fontweight='bold'); axes[1,1].grid(alpha=0.15)

    # Chart 6: Real Dynamic Phase Accuracy
    xp = np.arange(3)
    vb = axes[1,2].bar(xp - 0.15, v_ph, 0.3, label='Standard', color=v_c, alpha=0.8)
    gb = axes[1,2].bar(xp + 0.15, g_ph, 0.3, label='Geo-Llama', color=g_c, alpha=0.8)
    vb[1].set_edgecolor('black'); vb[1].set_linewidth(1.5)
    gb[1].set_edgecolor('black'); gb[1].set_linewidth(1.5)
    axes[1,2].set_xticks(xp); axes[1,2].set_xticklabels(['Ordered', 'Critical', 'Disordered'])
    axes[1,2].set_title('Accuracy per Physics Phase', fontweight='bold'); axes[1,2].set_ylim(40, 115)
    for i, (v, g) in enumerate(zip(v_ph, g_ph)):
        axes[1,2].text(i-0.15, v+1, f"{v:.1f}%", ha='center', fontsize=9, fontweight='bold')
        axes[1,2].text(i+0.15, g+1, f"{g:.1f}%", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_path = "/Users/mac/Desktop/Geo-llama/Research/tests/final_product_results.png"
    plt.savefig(save_path, dpi=200)
    print(f"\nFinal Results saved to: {save_path}")

if __name__ == "__main__":
    print(f"Using Device: {DEVICE}")
    run_master_benchmark()
