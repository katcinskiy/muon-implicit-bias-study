"""
Run comparison experiment between Muon and AdamW optimizers.
"""

from config import ExperimentConfig
from experiment import run_experiment


def main():
    """Run experiments with both optimizers for comparison."""

    print("\n" + "=" * 80)
    print("Running Comparison: Muon vs AdamW")
    print("=" * 80)

    # Experiment 1: Muon optimizer
    print("\n\n### EXPERIMENT 1: MUON OPTIMIZER ###\n")
    config_muon = ExperimentConfig(
        optimizer_type="muon",
        output_dir="./results/muon",
    )
    run_experiment(config_muon)

    # Experiment 2: AdamW optimizer
    print("\n\n### EXPERIMENT 2: ADAMW OPTIMIZER ###\n")
    config_adamw = ExperimentConfig(
        optimizer_type="adamw",
        output_dir="./results/adamw",
    )
    run_experiment(config_adamw)

    print("\n" + "=" * 80)
    print("Comparison Complete!")
    print("=" * 80)
    print("\nResults saved to:")
    print("  - ./results/muon/results_muon.json")
    print("  - ./results/adamw/results_adamw.json")
    print("\nYou can compare how each optimizer handles knowledge override.")
    print("=" * 80)


if __name__ == "__main__":
    main()
