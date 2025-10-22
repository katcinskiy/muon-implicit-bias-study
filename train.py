"""
Training utilities with Muon optimizer for LoRA and AdamW for biases.
"""

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from tqdm import tqdm
import os


def setup_optimizers(model: PreTrainedModel, config):
    """
    Set up optimizers: Muon for LoRA parameters, AdamW for biases.

    Args:
        model: The model with LoRA adapters
        config: Experiment configuration

    Returns:
        Tuple of (lora_optimizer, bias_optimizer) or single optimizer
    """
    # Separate LoRA parameters and bias parameters
    lora_params = []
    bias_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Bias parameters
        if "bias" in name:
            bias_params.append(param)
        # LoRA parameters (usually contain "lora" in the name)
        elif "lora" in name.lower():
            lora_params.append(param)
        else:
            # Other trainable parameters - treat as LoRA params for this experiment
            lora_params.append(param)

    print(f"LoRA parameters: {len(lora_params)}")
    print(f"Bias parameters: {len(bias_params)}")

    if config.optimizer_type == "muon":
        # Use Muon for LoRA parameters
        lora_optimizer = torch.optim.Muon(
            lora_params,
            lr=config.phase1_learning_rate,
            momentum=config.muon_momentum,
            backend=config.muon_backend,
        )
    else:
        # Use AdamW for LoRA parameters
        lora_optimizer = torch.optim.AdamW(
            lora_params,
            lr=config.phase1_learning_rate,
            betas=(config.adamw_beta1, config.adamw_beta2),
            weight_decay=config.adamw_weight_decay,
        )

    # Always use AdamW for biases
    bias_optimizer = None
    if len(bias_params) > 0:
        bias_optimizer = torch.optim.AdamW(
            bias_params,
            lr=config.phase1_learning_rate,
            betas=(config.adamw_beta1, config.adamw_beta2),
            weight_decay=config.adamw_weight_decay,
        )

    return lora_optimizer, bias_optimizer


def train_epoch(
    model: PreTrainedModel,
    dataloader: DataLoader,
    lora_optimizer: torch.optim.Optimizer,
    bias_optimizer: torch.optim.Optimizer | None,
    device: str,
    desc: str = "Training",
    phase: str = "phase1",
    global_step: int = 0,
    log_wandb: bool = True,
) -> tuple[float, int]:
    """
    Train for one epoch.

    Args:
        model: The model to train
        dataloader: DataLoader with training data
        lora_optimizer: Optimizer for LoRA parameters
        bias_optimizer: Optimizer for bias parameters (optional)
        device: Device to train on
        desc: Description for progress bar
        phase: Training phase name (for wandb logging)
        global_step: Current global step
        log_wandb: Whether to log to wandb

    Returns:
        Tuple of (average loss for the epoch, updated global_step)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=desc)

    for batch in progress_bar:
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Update LoRA parameters
        lora_optimizer.step()
        lora_optimizer.zero_grad()

        # Update bias parameters
        if bias_optimizer is not None:
            bias_optimizer.step()
            bias_optimizer.zero_grad()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        # Log to wandb
        if log_wandb:
            import wandb
            wandb.log({f"{phase}/loss": loss.item(), f"{phase}/step": global_step})

        global_step += 1
        progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / num_batches
    return avg_loss, global_step


def train_phase(
    model: PreTrainedModel,
    dataset,
    lora_optimizer: torch.optim.Optimizer,
    bias_optimizer: torch.optim.Optimizer | None,
    config,
    phase_name: str,
    num_epochs: int,
    batch_size: int,
    device: str,
    phase_key: str = "phase1",
    log_wandb: bool = True,
) -> list[float]:
    """
    Train for a complete phase.

    Args:
        model: The model to train
        dataset: Training dataset
        lora_optimizer: Optimizer for LoRA parameters
        bias_optimizer: Optimizer for bias parameters
        config: Experiment configuration
        phase_name: Name of the phase (for logging)
        num_epochs: Number of epochs to train
        batch_size: Batch size
        device: Device to train on
        phase_key: Phase key for wandb logging (phase1 or phase2)
        log_wandb: Whether to log to wandb

    Returns:
        List of average losses per epoch
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    global_step = 0

    for epoch in range(num_epochs):
        desc = f"{phase_name} - Epoch {epoch + 1}/{num_epochs}"
        avg_loss, global_step = train_epoch(
            model=model,
            dataloader=dataloader,
            lora_optimizer=lora_optimizer,
            bias_optimizer=bias_optimizer,
            device=device,
            desc=desc,
            phase=phase_key,
            global_step=global_step,
            log_wandb=log_wandb,
        )
        losses.append(avg_loss)
        print(f"{desc} - Avg Loss: {avg_loss:.4f}")

        # Log epoch metrics
        if log_wandb:
            import wandb
            wandb.log({f"{phase_key}/avg_loss": avg_loss, f"{phase_key}/epoch": epoch + 1})

    return losses


def save_checkpoint(model: PreTrainedModel, save_path: str):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
