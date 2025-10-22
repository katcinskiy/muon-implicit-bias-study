import wandb


def init_wandb(config, project_name: str = "lora-orthogonalization"):
    if isinstance(config, dict):
        optimizer_type = config["optimizer_type"]
        phase2_samples = config["phase2_samples"]
    else:
        optimizer_type = config.optimizer_type
        phase2_samples = config.phase2_samples

    run = wandb.init(
        project=project_name,
        name=f"{optimizer_type}_phase2_{phase2_samples}",
        config=config,
        tags=[optimizer_type, f"phase2_{phase2_samples}"],
    )
    return run


def log_metrics(metrics: dict, step: int = None):
    if step is not None:
        metrics["step"] = step
    wandb.log(metrics)
