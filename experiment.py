import torch
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import random
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import create_phase1_dataset, create_phase2_dataset
from train import setup_optimizers, train_phase, save_checkpoint
from evaluate import evaluate_both_facts
from wandb_utils import init_wandb, log_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("=" * 80)
    print("LoRA Orthogonalization Experiment")
    print("=" * 80)
    print(f"Optimizer: {cfg.optimizer_type}")
    print("=" * 80)

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    init_wandb(config_dict)

    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float32,
        device_map=device,
    )

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    lora_optimizer, bias_optimizer = setup_optimizers(model, cfg)

    phase1_dataset = create_phase1_dataset(cfg, tokenizer)
    phase2_dataset = create_phase2_dataset(cfg, tokenizer)

    print("\nInitial Evaluation")
    initial_probs = evaluate_both_facts(model, tokenizer, cfg, device)
    print(f"P(Paris): {initial_probs['prob_paris']:.6f}")
    print(f"P(Lyon): {initial_probs['prob_lyon']:.6f}")

    log_metrics({
        "prob_paris": initial_probs["prob_paris"],
        "prob_lyon": initial_probs["prob_lyon"],
        "prob_ratio": initial_probs["prob_lyon"] / (initial_probs["prob_paris"] + 1e-10),
        "stage": "initial",
    })

    print("\nPhase 1: Training on Original Fact")
    train_phase(
        model=model,
        dataset=phase1_dataset,
        lora_optimizer=lora_optimizer,
        bias_optimizer=bias_optimizer,
        config=cfg,
        phase_name="Phase 1",
        num_epochs=cfg.phase1_epochs,
        batch_size=cfg.phase1_batch_size,
        device=device,
        phase_key="phase1",
    )

    print("\nEvaluation after Phase 1")
    phase1_probs = evaluate_both_facts(model, tokenizer, cfg, device)
    print(f"P(Paris): {phase1_probs['prob_paris']:.6f}")
    print(f"P(Lyon): {phase1_probs['prob_lyon']:.6f}")

    log_metrics({
        "prob_paris": phase1_probs["prob_paris"],
        "prob_lyon": phase1_probs["prob_lyon"],
        "prob_ratio": phase1_probs["prob_lyon"] / (phase1_probs["prob_paris"] + 1e-10),
        "stage": "after_phase1",
    })

    checkpoint_path = os.path.join(cfg.output_dir, f"{cfg.optimizer_type}_phase1")
    save_checkpoint(model, checkpoint_path)

    print("\nPhase 2: Training on Conflicting Fact")
    train_phase(
        model=model,
        dataset=phase2_dataset,
        lora_optimizer=lora_optimizer,
        bias_optimizer=bias_optimizer,
        config=cfg,
        phase_name="Phase 2",
        num_epochs=cfg.phase2_epochs,
        batch_size=cfg.phase2_batch_size,
        device=device,
        phase_key="phase2",
    )

    print("\nFinal Evaluation")
    final_probs = evaluate_both_facts(model, tokenizer, cfg, device)
    print(f"P(Paris): {final_probs['prob_paris']:.6f}")
    print(f"P(Lyon): {final_probs['prob_lyon']:.6f}")

    log_metrics({
        "prob_paris": final_probs["prob_paris"],
        "prob_lyon": final_probs["prob_lyon"],
        "prob_ratio": final_probs["prob_lyon"] / (final_probs["prob_paris"] + 1e-10),
        "stage": "after_phase2",
    })

    final_checkpoint_path = os.path.join(cfg.output_dir, f"{cfg.optimizer_type}_final")
    save_checkpoint(model, final_checkpoint_path)

    results = {
        "config": config_dict,
        "initial": initial_probs,
        "after_phase1": phase1_probs,
        "after_phase2": final_probs,
    }

    results_path = os.path.join(cfg.output_dir, f"results_{cfg.optimizer_type}.json")
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print("\n" + "=" * 80)
    print("Experiment Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
