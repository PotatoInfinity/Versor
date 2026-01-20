# Neural Newton

A demonstration of **Geometric Algebra** for learning physical laws from data.
This project compares a **Standard Transformer** against a **Geo-Llama Rotor RNN** on an N-body gravity simulation.

## The Problem
Predicting the motion of N stars interacting via gravity ($F = G \frac{m_1 m_2}{r^2}$) requires respecting conservation laws (Energy, Momentum) and Symmetries (Rotation, Translation). Standard Neural Networks often violate these, causing planets to spiral into the sun or drift away.

## The Solution: Geo-Llama
By embedding the state into a **Geometric Algebra (Cl(4,1))** multivector and using **Rotor-based updates**, the model learns to respect geometric constraints (like rotations being structure-preserving) naturally.

## Contents
- `data_gen.py`: Generates synthetic N-body gravity data (positions, velocities).
- `models.py`:
    - **StandardTransformer**: A baseline `nn.TransformerEncoder`.
    - **GeometricRotorRNN**: A Recurrent Network using Geometric Linear Layers (`algebra.py`).
- `train.py`: Trains both models and compares them on **Energy Drift** and **MSE**.
- `algebra.py`: The Geometric Algebra kernel (from Geo-Llama).

## Usage
Run the training and comparison:
```bash
python3 Neural\ Newton/train.py
```

## Measured Results (30 Epochs)
Both models were trained on 200 trajectories of 5-body interaction.

| Model | MSE (Lower is Better) | Energy Drift (Lower is Better) |
|-------|-----------------------|--------------------------------|
| Standard Transformer | 23.56 | 229.23 |
| **GeoLlama Rotor RNN** | **5.79** | **74.03** |

**Conclusion**: 
The Geometric Algebra model significantly outperforms the Standard Transformer.
1. **O(1) Context**: It maintains state more effectively (~4x better MSE).
2. **Physics Compliance**: The Manifold Normalization and Geometric updates keep the energy drift ~3x lower than the baseline, meaning the simulation stays stable for longer.

## Requirements
- PyTorch
- NumPy
