import numpy as np
import torch
import os

def generate_ising_2d(size, temp, steps=50000):
    """
    Generates a 2D Ising grid (+1, -1) using the Metropolis-Hastings algorithm.
    Optimized for small-to-medium lattices.
    """
    grid = np.random.choice([1, -1], size=(size, size))
    for _ in range(steps):
        # Pick random site
        i, j = np.random.randint(0, size, 2)
        # Calculate energy change dE
        spin = grid[i, j]
        neighbor_sum = (
            grid[(i+1)%size, j] + grid[(i-1)%size, j] +
            grid[i, (j+1)%size] + grid[i, (j-1)%size]
        )
        dE = 2 * spin * neighbor_sum
        
        # Metropolis acceptance criterion
        if dE <= 0 or np.random.rand() < np.exp(-dE / temp):
            grid[i, j] *= -1
    return grid

def generate_dataset(n_samples_per_class, size=8):
    """
    Generates a dataset of Ising configurations for three distinct physical phases:
    - Ordered (T=1.0 < T_c): Low entropy, high magnetization.
    - Critical (T=2.269 = T_c): Fractal-like correlations, phase transition.
    - Disordered (T=5.0 > T_c): High entropy, random spins.
    """
    temps = [1.0, 2.269, 5.0]
    data = []
    labels = []
    
    for label, T in enumerate(temps):
        print(f"Generating Phase {label} (T={T})...")
        for _ in range(n_samples_per_class):
            grid = generate_ising_2d(size, T)
            data.append(grid)
            labels.append(label)
            
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    return data, labels

if __name__ == "__main__":
    # Settings for a publishable-grade benchmark
    SAMPLES_PER_CLASS = 200 # Total 600 samples
    GRID_SIZE = 8  # 8x8 grid
    DATA_DIR = "/Users/mac/Desktop/Geo-llama/Research/data"
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print(f"--- Ising Data Generation (Size={GRID_SIZE}x{GRID_SIZE}) ---")
    data, labels = generate_dataset(n_samples_per_class=SAMPLES_PER_CLASS, size=GRID_SIZE)
    
    data_path = os.path.join(DATA_DIR, "ising_data.pt")
    torch.save({
        'data': torch.from_numpy(data),
        'labels': torch.from_numpy(labels),
        'metadata': {
            'size': GRID_SIZE,
            'temps': [1.0, 2.269, 5.0],
            'steps_per_sample': 50000
        }
    }, data_path)
    
    print(f"Successfully generated {len(data)} samples.")
    print(f"Saved to: {data_path}")


