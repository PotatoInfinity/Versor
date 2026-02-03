#!/usr/bin/env python3
"""
Master Experiment Runner for Versor Paper
Runs all experiments and saves results with timestamps
"""

import json
import os
import sys
import time
from datetime import datetime
import numpy as np
import torch

# Create results directory
RESULTS_DIR = "paper_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_results(experiment_name, results_dict):
    """Save results with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{RESULTS_DIR}/{experiment_name}_{timestamp}.json"
    
    # Add metadata
    results_dict["metadata"] = {
        "timestamp": timestamp,
        "experiment": experiment_name,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "CPU"
    }
    
    with open(filename, 'w') as f:
        json.dump(results_dict, indent=2, fp=f)
    
    print(f"✓ Saved: {filename}")
    return filename

def run_experiment_1_nbody():
    """N-Body Physics Experiment (Table 2 in paper)"""
    print("\n" + "="*60)
    print("EXPERIMENT 1: N-Body Dynamics")
    print("="*60)
    
    sys.path.append("Physics")
    from train import train
    from models import StandardTransformer, VersorRotorRNN, GraphNetworkSimulator, HamiltonianNN
    from data_gen import generate_gravity_data
    
    # TODO: Actually run the training
    # This is a placeholder - you need to modify train.py to return results
    
    results = {
        "experiment": "nbody_dynamics",
        "models": {
            "transformer": {"mse": None, "energy_drift": None},
            "versor": {"mse": None, "energy_drift": None},
            "gns": {"mse": None, "energy_drift": None},
            "hnn": {"mse": None, "energy_drift": None}
        },
        "config": {
            "n_particles": 5,
            "train_samples": 200,
            "test_samples": 10,
            "rollout_steps": 100,
            "seed": 42
        }
    }
    
    print("⚠️  WARNING: You need to modify Physics/train.py to return results")
    print("⚠️  This is a template - fill in actual numbers")
    
    return save_results("nbody", results)

def run_experiment_2_topology():
    """Maze Connectivity (Broken Snake)"""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Topological Connectivity")
    print("="*60)
    
    sys.path.append("Maze")
    
    results = {
        "experiment": "topology_maze",
        "grid_sizes": {},
        "config": {
            "train_grids": [8, 16],
            "test_grids": [16, 32],
            "curriculum": True
        }
    }
    
    # Run for different grid sizes
    for size in [8, 16, 32]:
        print(f"\nTesting grid size: {size}x{size}")
        # TODO: Import and run your maze experiments
        results["grid_sizes"][str(size)] = {
            "versor_mcc": None,
            "std_mcc": None,
            "vit_mcc": None
        }
    
    print("⚠️  WARNING: You need to run Maze experiments")
    
    return save_results("topology", results)

def run_experiment_3_ood():
    """Out-of-Distribution Generalization"""
    print("\n" + "="*60)
    print("EXPERIMENT 3: OOD Generalization (Heavy Masses)")
    print("="*60)
    
    try:
        from Physics import recreate_ood
        recreate_ood.run_ood_test()
        
        # Load the results it barely saved
        with open('results/ood_mass_results.json', 'r') as f:
            data = json.load(f)
            
        print("\n✓ OOD Experiment Completed Successfully")
        return save_results("ood", data)
        
    except ImportError as e:
        print(f"❌ Error importing OOD module: {e}")
        return None
    except Exception as e:
        print(f"❌ Error running OOD experiment: {e}")
        return None

def run_experiment_4_ablation():
    """Ablation Study (Table, Line 594)"""
    print("\n" + "="*60)
    print("EXPERIMENT 4: Ablation Study")
    print("="*60)
    
    try:
        from Physics import rigorous_ablation
        rigorous_ablation.main()
        
        # Load the results it saved
        with open('results/ablation_stats.json', 'r') as f:
            data = json.load(f)
            
        print("\n✓ Ablation Study Completed Successfully")
        return save_results("ablation", data)
        
    except ImportError as e:
        print(f"❌ Error importing Ablation module: {e}")
        return None
    except Exception as e:
        print(f"❌ Error running Ablation experiment: {e}")
        return None

def run_experiment_5_kernel_benchmark():
    """Kernel Performance Benchmark"""
    print("\n" + "="*60)
    print("EXPERIMENT 5: Kernel Performance")
    print("="*60)
    
    import kernel
    
    if not kernel.HAS_TRITON:
        print("❌ Triton not available - skipping GPU benchmark")
        return None
    
    # Test different batch sizes
    batch_sizes = [16, 32, 64, 128, 256, 512, 1024]
    
    results = {
        "experiment": "kernel_benchmark",
        "batch_sizes": {},
        "implementations": ["pytorch_naive", "pytorch_einsum", "triton"]
    }
    
    print("⚠️  WARNING: Need to implement proper benchmarking")
    
    return save_results("kernel_bench", results)

def run_multi_seed_validation():
    """Run experiments with multiple seeds for statistical validity"""
    print("\n" + "="*60)
    print("STATISTICAL VALIDATION: Multiple Seeds")
    print("="*60)
    
    seeds = [42, 123, 456, 789, 2024]
    
    results = {
        "experiment": "multi_seed_validation",
        "seeds": seeds,
        "runs": []
    }
    
    for seed in seeds:
        print(f"\nSeed: {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # TODO: Run main experiment with this seed
        run_results = {
            "seed": seed,
            "mse": None,
            "energy_drift": None
        }
        results["runs"].append(run_results)
    
    # Calculate statistics
    if results["runs"]:
        mses = [r["mse"] for r in results["runs"] if r["mse"] is not None]
        if mses:
            results["statistics"] = {
                "mse_mean": float(np.mean(mses)),
                "mse_std": float(np.std(mses)),
                "mse_min": float(np.min(mses)),
                "mse_max": float(np.max(mses))
            }
    
    print("⚠️  WARNING: Need to integrate with actual training")
    
    return save_results("multi_seed", results)

def generate_summary_report():
    """Generate a summary of all results"""
    print("\n" + "="*60)
    print("GENERATING SUMMARY REPORT")
    print("="*60)
    
    # Find all result files
    result_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')]
    
    if not result_files:
        print("❌ No results found!")
        return
    
    print(f"\nFound {len(result_files)} result files:")
    for f in sorted(result_files):
        print(f"  - {f}")
    
    # Load and summarize
    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_experiments": len(result_files),
        "experiments": {}
    }
    
    for fname in result_files:
        with open(os.path.join(RESULTS_DIR, fname)) as f:
            data = json.load(f)
            exp_name = data.get("metadata", {}).get("experiment", "unknown")
            summary["experiments"][exp_name] = {
                "file": fname,
                "timestamp": data.get("metadata", {}).get("timestamp"),
                "status": "completed" if data else "empty"
            }
    
    summary_file = f"{RESULTS_DIR}/SUMMARY_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, indent=2, fp=f)
    
    print(f"\n✓ Summary saved to: {summary_file}")

if __name__ == "__main__":
    print("="*60)
    print("VERSOR PAPER - COMPLETE EXPERIMENTAL SUITE")
    print("="*60)
    print(f"Started at: {datetime.now()}")
    print(f"Results will be saved to: {RESULTS_DIR}/")
    print("="*60)
    
    # Run all experiments
    experiments = [
        run_experiment_1_nbody,
        run_experiment_2_topology,
        run_experiment_3_ood,
        run_experiment_4_ablation,
        run_experiment_5_kernel_benchmark,
        run_multi_seed_validation
    ]
    
    for exp_func in experiments:
        try:
            exp_func()
        except Exception as e:
            print(f"\n❌ ERROR in {exp_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate summary
    generate_summary_report()
    
    print("\n" + "="*60)
    print("EXPERIMENT SUITE COMPLETE")
    print("="*60)
    print(f"\n⚠️  IMPORTANT: Many experiments are templates only!")
    print("You need to:")
    print("1. Implement the missing experiment code")
    print("2. Run each experiment properly")
    print("3. Verify results match paper claims")
    print("4. Add error bars from multiple runs")
    print(f"\nFinished at: {datetime.now()}")
