import wandb
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def init_wandb(config, project_name: str = "lora-orthogonalization", run_name: Optional[str] = None):
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


def log_evaluation(prob_paris: float, prob_lyon: float, stage: str, step: int = 0):
    wandb.log({
        f"eval/{stage}/prob_paris": prob_paris,
        f"eval/{stage}/prob_lyon": prob_lyon,
        f"eval/{stage}/prob_ratio": prob_lyon / (prob_paris + 1e-10),
        "step": step,
    })


def create_probability_evolution_plot(evaluations: list[dict]) -> plt.Figure:
    stages = [eval_dict["stage"] for eval_dict in evaluations]
    prob_paris = [eval_dict["prob_paris"] for eval_dict in evaluations]
    prob_lyon = [eval_dict["prob_lyon"] for eval_dict in evaluations]

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(range(1, len(phase1_losses) + 1), phase1_losses, marker="o", color="#3498db", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Average Loss", fontsize=12, fontweight="bold")
    ax1.set_title("Phase 1: Original Fact (Paris)", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, len(phase2_losses) + 1), phase2_losses, marker="o", color="#e74c3c", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Average Loss", fontsize=12, fontweight="bold")
    ax2.set_title("Phase 2: Conflicting Fact (Lyon)", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_knowledge_override_plot(evaluations: list[dict]) -> plt.Figure:
    stages = [eval_dict["stage"] for eval_dict in evaluations]
    prob_paris = [eval_dict["prob_paris"] for eval_dict in evaluations]
    prob_lyon = [eval_dict["prob_lyon"] for eval_dict in evaluations]

    ratios = [lyon / (paris + 1e-10) for paris, lyon in zip(prob_paris, prob_lyon)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(stages, ratios, marker="o", linewidth=2.5, markersize=10, color="#9b59b6")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Equal probability")

    ax.set_xlabel("Stage", fontsize=12, fontweight="bold")
    ax.set_ylabel("P(Lyon) / P(Paris)", fontsize=12, fontweight="bold")
    ax.set_title("Knowledge Override Effect", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.xticks(rotation=15, ha="right")

    plt.tight_layout()
    return fig


def log_plots(evaluations: list[dict], phase1_losses: list[float], phase2_losses: list[float]):
    prob_fig = create_probability_evolution_plot(evaluations)
    wandb.log({"plots/probability_evolution": wandb.Image(prob_fig)})
    plt.close(prob_fig)

    loss_fig = create_loss_plot(phase1_losses, phase2_losses)
    wandb.log({"plots/training_losses": wandb.Image(loss_fig)})
    plt.close(loss_fig)

    override_fig = create_knowledge_override_plot(evaluations)
    wandb.log({"plots/knowledge_override": wandb.Image(override_fig)})
    plt.close(override_fig)


def log_summary_table(evaluations: list[dict]):
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
