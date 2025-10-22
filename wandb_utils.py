"""
Weights & Biases logging utilities and plotting functions.
"""

import wandb
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def init_wandb(config, project_name: str = "lora-orthogonalization", run_name: Optional[str] = None):
    """
    Initialize Weights & Biases run.

    Args:
        config: Experiment configuration
        project_name: Name of the wandb project
        run_name: Optional custom run name

    Returns:
        wandb run object
    """
    if run_name is None:
        run_name = f"{config.optimizer_type}_phase2_{config.phase2_samples}"

    run = wandb.init(
        project=project_name,
        name=run_name,
        config={
            "model_name": config.model_name,
            "optimizer_type": config.optimizer_type,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "phase1_samples": config.phase1_samples,
            "phase1_epochs": config.phase1_epochs,
            "phase1_batch_size": config.phase1_batch_size,
            "phase1_learning_rate": config.phase1_learning_rate,
            "phase2_samples": config.phase2_samples,
            "phase2_epochs": config.phase2_epochs,
            "phase2_batch_size": config.phase2_batch_size,
            "phase2_learning_rate": config.phase2_learning_rate,
            "original_fact": config.original_fact,
            "conflicting_fact": config.conflicting_fact,
            "seed": config.seed,
        },
        tags=[config.optimizer_type, f"phase2_{config.phase2_samples}"],
    )

    return run


def log_training_step(loss: float, step: int, phase: str):
    """
    Log training step metrics to wandb.

    Args:
        loss: Training loss
        step: Global step number
        phase: Training phase (phase1 or phase2)
    """
    wandb.log({f"{phase}/loss": loss, f"{phase}/step": step})


def log_epoch_metrics(avg_loss: float, epoch: int, phase: str):
    """
    Log epoch-level metrics to wandb.

    Args:
        avg_loss: Average loss for the epoch
        epoch: Epoch number
        phase: Training phase (phase1 or phase2)
    """
    wandb.log({f"{phase}/avg_loss": avg_loss, f"{phase}/epoch": epoch})


def log_evaluation(prob_paris: float, prob_lyon: float, stage: str, step: int = 0):
    """
    Log evaluation metrics to wandb.

    Args:
        prob_paris: Probability of "Paris"
        prob_lyon: Probability of "Lyon"
        stage: Evaluation stage (initial, after_phase1, after_phase2)
        step: Global step number
    """
    wandb.log(
        {
            f"eval/{stage}/prob_paris": prob_paris,
            f"eval/{stage}/prob_lyon": prob_lyon,
            f"eval/{stage}/prob_ratio": prob_lyon / (prob_paris + 1e-10),
            "step": step,
        }
    )


def create_probability_evolution_plot(evaluations: list[dict]) -> plt.Figure:
    """
    Create a plot showing probability evolution across experiment stages.

    Args:
        evaluations: List of evaluation dictionaries with prob_paris and prob_lyon

    Returns:
        Matplotlib figure
    """
    stages = [eval_dict["stage"] for eval_dict in evaluations]
    prob_paris = [eval_dict["prob_paris"] for eval_dict in evaluations]
    prob_lyon = [eval_dict["prob_lyon"] for eval_dict in evaluations]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(stages))
    width = 0.35

    bars1 = ax.bar(x - width / 2, prob_paris, width, label="P(Paris)", color="#2ecc71")
    bars2 = ax.bar(x + width / 2, prob_lyon, width, label="P(Lyon)", color="#e74c3c")

    ax.set_xlabel("Stage", fontsize=12, fontweight="bold")
    ax.set_ylabel("Probability", fontsize=12, fontweight="bold")
    ax.set_title("Probability Evolution: Paris vs Lyon", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    return fig


def create_loss_plot(phase1_losses: list[float], phase2_losses: list[float]) -> plt.Figure:
    """
    Create a plot showing training loss across both phases.

    Args:
        phase1_losses: Losses from phase 1
        phase2_losses: Losses from phase 2

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Phase 1 losses
    ax1.plot(range(1, len(phase1_losses) + 1), phase1_losses, marker="o", color="#3498db", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Average Loss", fontsize=12, fontweight="bold")
    ax1.set_title("Phase 1: Original Fact (Paris)", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Phase 2 losses
    ax2.plot(range(1, len(phase2_losses) + 1), phase2_losses, marker="o", color="#e74c3c", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Average Loss", fontsize=12, fontweight="bold")
    ax2.set_title("Phase 2: Conflicting Fact (Lyon)", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_knowledge_override_plot(evaluations: list[dict]) -> plt.Figure:
    """
    Create a plot showing the knowledge override effect (Lyon/Paris ratio).

    Args:
        evaluations: List of evaluation dictionaries

    Returns:
        Matplotlib figure
    """
    stages = [eval_dict["stage"] for eval_dict in evaluations]
    prob_paris = [eval_dict["prob_paris"] for eval_dict in evaluations]
    prob_lyon = [eval_dict["prob_lyon"] for eval_dict in evaluations]

    # Calculate ratio (with small epsilon to avoid division by zero)
    ratios = [lyon / (paris + 1e-10) for paris, lyon in zip(prob_paris, prob_lyon)]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(stages, ratios, marker="o", linewidth=2.5, markersize=10, color="#9b59b6")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Equal probability")

    ax.set_xlabel("Stage", fontsize=12, fontweight="bold")
    ax.set_ylabel("P(Lyon) / P(Paris)", fontsize=12, fontweight="bold")
    ax.set_title("Knowledge Override Effect", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Rotate x-axis labels
    plt.xticks(rotation=15, ha="right")

    plt.tight_layout()
    return fig


def log_plots(evaluations: list[dict], phase1_losses: list[float], phase2_losses: list[float]):
    """
    Create and log all plots to wandb.

    Args:
        evaluations: List of evaluation dictionaries
        phase1_losses: Losses from phase 1
        phase2_losses: Losses from phase 2
    """
    # Probability evolution plot
    prob_fig = create_probability_evolution_plot(evaluations)
    wandb.log({"plots/probability_evolution": wandb.Image(prob_fig)})
    plt.close(prob_fig)

    # Loss plot
    loss_fig = create_loss_plot(phase1_losses, phase2_losses)
    wandb.log({"plots/training_losses": wandb.Image(loss_fig)})
    plt.close(loss_fig)

    # Knowledge override plot
    override_fig = create_knowledge_override_plot(evaluations)
    wandb.log({"plots/knowledge_override": wandb.Image(override_fig)})
    plt.close(override_fig)


def log_summary_table(evaluations: list[dict]):
    """
    Create and log a summary table to wandb.

    Args:
        evaluations: List of evaluation dictionaries
    """
    table_data = []
    for eval_dict in evaluations:
        stage = eval_dict["stage"]
        prob_paris = eval_dict["prob_paris"]
        prob_lyon = eval_dict["prob_lyon"]
        ratio = prob_lyon / (prob_paris + 1e-10)
        table_data.append([stage, prob_paris, prob_lyon, ratio])

    table = wandb.Table(
        columns=["Stage", "P(Paris)", "P(Lyon)", "P(Lyon)/P(Paris)"],
        data=table_data,
    )
    wandb.log({"summary/evaluation_table": table})
