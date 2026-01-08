import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import time
#torch.manual_seed(27)
#np.random.seed(27) # For reproducibility
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geo_transformer.core import conformal_lift
from geo_transformer.model import GeometricTransformer

# Comparison Hyperparameters
EMBED_DIM = 8
N_HEADS = 2
N_LAYERS = 1
BATCH_SIZE = 16
EPOCHS = 50
LR = 0.001
N_CLASSES = 3

class VanillaTransformer(nn.Module):
    """ Standard PyTorch Transformer for comparison. """
    def __init__(self, input_dim, d_model, nhead, n_layers, dim_feedforward, n_classes):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, n_classes)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1) 
        return self.classifier(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, train_loader, val_loader, model_name):
    print(f"\n--- Training {model_name} ---")
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    start_time = time.time()
    
    best_val_acc = -1.0
    best_state = None
    epochs_no_improve = 0
    patience = 10
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            # Reshape for Vanilla if needed
            if "Vanilla" in model_name:
                x = x.view(x.shape[0], x.shape[1], -1)
            
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            correct += (torch.argmax(logits, 1) == y).sum().item()
            total += y.size(0)
            
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation
        model.eval()
        v_loss, v_corr, v_total = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                if "Vanilla" in model_name: 
                    x = x.view(x.shape[0], x.shape[1], -1)
                logits = model(x)
                v_loss += criterion(logits, y).item()
                v_corr += (torch.argmax(logits, 1) == y).sum().item()
                v_total += y.size(0)
        
        val_loss = v_loss / len(val_loader)
        val_acc = v_corr / v_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Update Best State
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Best: {best_val_acc:.4f}")
        
        if epochs_no_improve >= patience:
            print(f"--- Early stopping triggered for {model_name} after {epoch+1} epochs ---")
            break
        
    # Restore Best State
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"--- Best State restored for {model_name} (Acc: {best_val_acc:.4f}) ---")
        
    elapsed = time.time() - start_time
    print(f"{model_name} finished in {elapsed:.2f}s")
    return history

def evaluate_detailed(model, loader, model_name):
    model.eval()
    class_correct = [0] * N_CLASSES
    class_total = [0] * N_CLASSES
    with torch.no_grad():
        for x, y in loader:
            if "Vanilla" in model_name: x = x.view(x.shape[0], x.shape[1], -1)
            logits = model(x)
            preds = torch.argmax(logits, 1)
            for i in range(len(y)):
                label = y[i].item()
                class_total[label] += 1
                if preds[i] == label: class_correct[label] += 1
    return [class_correct[i]/class_total[i] if class_total[i] > 0 else 0 for i in range(N_CLASSES)]

def run_benchmark():
    data_path = "/Users/mac/Desktop/Geo-llama/Research/data/ising_data.pt"
    if not os.path.exists(data_path):
        print("Data not found. Run Research/ising/data_gen.py first.")
        return
        
    checkpoint = torch.load(data_path)
    X_raw, Y = checkpoint['data'], checkpoint['labels']
    
    X_lifted = conformal_lift(X_raw.view(X_raw.shape[0], -1)) 
    X_input = X_lifted.unsqueeze(2).repeat(1, 1, EMBED_DIM, 1) 
    
    full_dataset = TensorDataset(X_input, Y)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    # 1. Geo-Llama (Mean Pool)
    geo_mean = GeometricTransformer(EMBED_DIM, N_HEADS, N_LAYERS, N_CLASSES, use_rotor_pool=False)
    geo_params = count_parameters(geo_mean)
    
    # 2. Geo-Llama (Rotor Pool)
    geo_rotor = GeometricTransformer(EMBED_DIM, N_HEADS, N_LAYERS, N_CLASSES, use_rotor_pool=True)
    
    # 3. Vanilla (Parameter Matched)
    # Target approx geo_params
    vanilla = VanillaTransformer(EMBED_DIM*32, 64, N_HEADS, N_LAYERS, 256, N_CLASSES)
    # Fine-tune ff to match closely
    base = count_parameters(VanillaTransformer(EMBED_DIM*32, 64, N_HEADS, N_LAYERS, 1, N_CLASSES))
    target_ff = int((geo_params - base) / (2 * 64))
    vanilla = VanillaTransformer(EMBED_DIM*32, 64, N_HEADS, N_LAYERS, max(1, target_ff), N_CLASSES)
    
    print(f"Parameters:")
    print(f" - Geo-Llama: {geo_params}")
    print(f" - Vanilla:   {count_parameters(vanilla)}")
    
    # Train all
    h_v = train_model(vanilla, train_loader, val_loader, "Vanilla")
    h_gm = train_model(geo_mean, train_loader, val_loader, "Geo-Mean")
    h_gr = train_model(geo_rotor, train_loader, val_loader, "Geo-Rotor")
    
    # Evaluate
    acc_v = evaluate_detailed(vanilla, val_loader, "Vanilla")
    acc_gm = evaluate_detailed(geo_mean, val_loader, "Geo-Mean")
    acc_gr = evaluate_detailed(geo_rotor, val_loader, "Geo-Rotor")
    
    # Plotting
    phases = ['Ordered', 'Critical', 'Disordered']
    plt.style.use('seaborn-v0_8-muted')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Losses
    ax1.plot(h_v['val_loss'], label='Vanilla', linestyle='--')
    ax1.plot(h_gm['val_loss'], label='Geo (Mean)', marker='s', markersize=4)
    ax1.plot(h_gr['val_loss'], label='Geo (Rotor)', marker='o', markersize=4)
    ax1.set_title('Validation Loss comparison')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Per-Class Accuracy
    x = np.arange(len(phases))
    width = 0.25
    ax2.bar(x - width, acc_v, width, label='Vanilla', alpha=0.8)
    ax2.bar(x, acc_gm, width, label='Geo (Mean)', alpha=0.8)
    ax2.bar(x + width, acc_gr, width, label='Geo (Rotor)', alpha=0.8)
    ax2.set_title('Accuracy by Phase')
    ax2.set_xticks(x, phases)
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("/Users/mac/Desktop/Geo-llama/Research/tests/benchmark_results.png", dpi=200)
    print("\nBenchmark Plot updated.")

if __name__ == "__main__":
    run_benchmark()
