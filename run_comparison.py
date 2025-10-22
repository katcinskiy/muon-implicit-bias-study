"""
Run comparison experiment between Muon and AdamW optimizers.
"""

import argparse
from config import ExperimentConfig
from experiment import run_experiment


def main(use_wandb: bool = True):
    """
    Run experiments with both optimizers for comparison.

    Args:
        use_wandb: Whether to use Weights & Biases for logging
    """

    print("\n" + "=" * 80)
    print("Running Comparison: Muon vs AdamW")
    print("=" * 80)

    # Experiment 1: Muon optimizer
    print("\n\n### EXPERIMENT 1: MUON OPTIMIZER ###\n")
    config_muon = ExperimentConfig(
        optimizer_type="muon",
        output_dir="./results/muon",
    )
    run_experiment(config_muon, use_wandb=use_wandb)

    # Experiment 2: AdamW optimizer
    print("\n\n### EXPERIMENT 2: ADAMW OPTIMIZER ###\n")
    config_adamw = ExperimentConfig(
        optimizer_type="adamw",
        output_dir="./results/adamw",
    )
    run_experiment(config_adamw, use_wandb=use_wandb)

    print("\n" + "=" * 80)
    print("Comparison Complete!")
    print("=" * 80)
    print("\nResults saved to:")
    print("  - ./results/muon/results_muon.json")
    print("  - ./results/adamw/results_adamw.json")
    if use_wandb:
        print("\nCheck your Weights & Biases dashboard to compare runs!")
    print("\nYou can compare how each optimizer handles knowledge override.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comparison between Muon and AdamW")
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    args = parser.parse_args()

    main(use_wandb=not args.no_wandb)
