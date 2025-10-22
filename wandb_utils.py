import wandb


def init_wandb(config, project_name: str = "lora-orthogonalization"):
    run = wandb.init(
        project=project_name,
        name=f"{config.optimizer_type}_phase2_{config.phase2_samples}",
        config=config,
        tags=[config.optimizer_type, f"phase2_{config.phase2_samples}"],
    )
    return run


def log_metrics(metrics: dict, step: int = None):
    if step is not None:
        metrics["step"] = step
    wandb.log(metrics)
