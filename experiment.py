"""
Main experiment runner for the LoRA orthogonalization study.
"""

import torch
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import random
import numpy as np

from config import ExperimentConfig
from dataset import create_phase1_dataset, create_phase2_dataset
from train import setup_optimizers, train_phase, save_checkpoint
from evaluate import evaluate_both_facts


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(config: ExperimentConfig):
    """
    Run the complete experiment.

    Args:
        config: Experiment configuration
    """
    print("=" * 80)
    print("LoRA Orthogonalization Experiment")
    print("=" * 80)
    print(f"Model: {config.model_name}")
    print(f"Optimizer: {config.optimizer_type}")
    print(f"Phase 1: {config.phase1_samples} samples of '{config.original_fact}'")
    print(f"Phase 2: {config.phase2_samples} samples of '{config.conflicting_fact}'")
    print("=" * 80)

    # Set seed
    set_seed(config.seed)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"\nLoading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32,  # Use float32 for stable training
        device_map=device,
    )

    # Apply LoRA
    print("\nApplying LoRA configuration...")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",  # Don't train biases via LoRA
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Setup optimizers
    print("\nSetting up optimizers...")
    lora_optimizer, bias_optimizer = setup_optimizers(model, config)

    # Create datasets
    print("\nCreating datasets...")
    phase1_dataset = create_phase1_dataset(config, tokenizer)
    phase2_dataset = create_phase2_dataset(config, tokenizer)

    # Results tracking
    results = {
        "config": config.__dict__,
        "evaluations": [],
    }

    # Initial evaluation
    print("\n" + "=" * 80)
    print("Initial Evaluation")
    print("=" * 80)
    initial_probs = evaluate_both_facts(model, tokenizer, config, device)
    print(f"P(Paris): {initial_probs['prob_paris']:.6f}")
    print(f"P(Lyon): {initial_probs['prob_lyon']:.6f}")
    results["evaluations"].append(
        {
            "stage": "initial",
            "prob_paris": initial_probs["prob_paris"],
            "prob_lyon": initial_probs["prob_lyon"],
        }
    )

    # Phase 1: Train on original fact
    print("\n" + "=" * 80)
    print("Phase 1: Training on Original Fact")
    print("=" * 80)
    phase1_losses = train_phase(
        model=model,
        dataset=phase1_dataset,
        lora_optimizer=lora_optimizer,
        bias_optimizer=bias_optimizer,
        config=config,
        phase_name="Phase 1",
        num_epochs=config.phase1_epochs,
        batch_size=config.phase1_batch_size,
        device=device,
    )

    # Evaluate after Phase 1
    print("\n" + "=" * 80)
    print("Evaluation after Phase 1")
    print("=" * 80)
    phase1_probs = evaluate_both_facts(model, tokenizer, config, device)
    print(f"P(Paris): {phase1_probs['prob_paris']:.6f}")
    print(f"P(Lyon): {phase1_probs['prob_lyon']:.6f}")
    results["evaluations"].append(
        {
            "stage": "after_phase1",
            "prob_paris": phase1_probs["prob_paris"],
            "prob_lyon": phase1_probs["prob_lyon"],
            "losses": phase1_losses,
        }
    )

    # Save checkpoint after Phase 1
    checkpoint_path = os.path.join(config.output_dir, f"{config.optimizer_type}_phase1")
    save_checkpoint(model, checkpoint_path)

    # Phase 2: Train on conflicting fact
    print("\n" + "=" * 80)
    print("Phase 2: Training on Conflicting Fact")
    print("=" * 80)
    phase2_losses = train_phase(
        model=model,
        dataset=phase2_dataset,
        lora_optimizer=lora_optimizer,
        bias_optimizer=bias_optimizer,
        config=config,
        phase_name="Phase 2",
        num_epochs=config.phase2_epochs,
        batch_size=config.phase2_batch_size,
        device=device,
    )

    # Final evaluation
    print("\n" + "=" * 80)
    print("Final Evaluation after Phase 2")
    print("=" * 80)
    final_probs = evaluate_both_facts(model, tokenizer, config, device)
    print(f"P(Paris): {final_probs['prob_paris']:.6f}")
    print(f"P(Lyon): {final_probs['prob_lyon']:.6f}")
    results["evaluations"].append(
        {
            "stage": "after_phase2",
            "prob_paris": final_probs["prob_paris"],
            "prob_lyon": final_probs["prob_lyon"],
            "losses": phase2_losses,
        }
    )

    # Save final checkpoint
    final_checkpoint_path = os.path.join(
        config.output_dir, f"{config.optimizer_type}_final"
    )
    save_checkpoint(model, final_checkpoint_path)

    # Save results
    results_path = os.path.join(
        config.output_dir, f"results_{config.optimizer_type}.json"
    )
    os.makedirs(config.output_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Summary
    print("\n" + "=" * 80)
    print("Experiment Summary")
    print("=" * 80)
    print(f"Optimizer: {config.optimizer_type}")
    print(f"\nProbability Changes:")
    print(f"  Initial:      P(Paris)={initial_probs['prob_paris']:.6f}, P(Lyon)={initial_probs['prob_lyon']:.6f}")
    print(f"  After Phase 1: P(Paris)={phase1_probs['prob_paris']:.6f}, P(Lyon)={phase1_probs['prob_lyon']:.6f}")
    print(f"  After Phase 2: P(Paris)={final_probs['prob_paris']:.6f}, P(Lyon)={final_probs['prob_lyon']:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    # Run with Muon optimizer
    config_muon = ExperimentConfig(optimizer_type="muon")
    run_experiment(config_muon)

    # Optionally run with AdamW for comparison
    # config_adamw = ExperimentConfig(optimizer_type="adamw")
    # run_experiment(config_adamw)
