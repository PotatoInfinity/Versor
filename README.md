# Versor: Foundational Theory of Structural Intelligence

Reference implementation for the paper **"Versor: Foundational Theory of Structural Intelligence"**. 

Versor is a sequence architecture built on **Conformal Geometric Algebra (CGA)** that replaces the "vector-space" assumptions of Transformers with the graded manifold structure of $Cl_{4,1}$. It solves the **Euclidean Bottleneck** by strictly enforcing $SE(3)$ symmetries through algebraic constraints.

## Repository Structure

- `Model/`: Core architecture components (`VersorLinear`, `GeometricProductAttention`).
- `Physics/`: N-Body dynamics experiments, data generation, and OOD benchmarks.
- `Maze/`: Topological connectivity tasks.
- `kernel.py`: Custom hardware-accelerated Clifford Algebra kernels (Triton/MLX).
- `run_all_experiments.py`: **Master script** to reproduce paper results.

## Installation

System dependencies: Python 3.9+

```bash
# Clone the repository
git clone https://github.com/PotatoInfinity/Versor.git
cd Versor

# Install dependencies
pip install -r requirements.txt
```

**Note on Hardware Acceleration:**
- **Linux (NVIDIA GPUs):** The codebase automatically leverages `triton` kernels for geometric products.
- **macOS (Apple Silicon):** Ensure `mlx` is installed for accelerated operations (experimental support).

## 🔬 Reproducing Results

We provide a master script to run the full experimental suite described in the paper.

### Quick Start (Full Suite)
```bash
python3 run_all_experiments.py
```
This script will:
1.  Train standard and Versor models on N-Body Dynamics.
2.  Run Topological Connectivity benchmarks.
3.  Execute the **OOD Mass Generalization** test (Paper Sec 7.2).
4.  Perform the **Ablation Study** (Manifold Norm, Recursive Rotor).
5.  Save all results to `./paper_results/`.

### Individual Experiments

**1. Verification of Initialization Strategy (Appendix G)**
Confirm that signal variance is preserved across 20 layers:
```bash
python3 verify_initialization.py
```

**2. Out-of-Distribution Generalization**
Train on standard masses ($m \in [0.5, 1.5]$) and test on heavy masses ($m \in [5.0, 10.0]$):
```bash
python3 Physics/recreate_ood.py
```

**3. Ablation Study**
Compare Full Versor vs. No-Norm vs. Standard Transformer:
```bash
python3 Physics/rigorous_ablation.py
```

## Reproducibility Statement

As noted in the paper:
> The provided minimal example code uses simplified hyperparameters (Learning Rate = $10^{-3}$, constant schedule) for rapid verification and CI/CD compatibility. 
>
> The **State-of-the-Art (SOTA)** results reported in the paper (Table 2) were obtained using the tuned schedule described in **Appendix L** (Cosine annealing, warmup, longer training horizon).

## Citation

If you use Versor in your research, please cite:

```bibtex
@article{versor2025,
  title={Versor: Foundational Theory of Structural Intelligence},
  author={Versor Team},
  journal={arXiv preprint},
  year={2026},
  doi={10.5281/zenodo.18320794},
  url={https://github.com/PotatoInfinity/Versor}
}
```

## License
MIT License
