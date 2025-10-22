import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from tqdm import tqdm
import os


def setup_optimizers(model: PreTrainedModel, config):
    lora_params = []
    bias_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name:
            bias_params.append(param)
        elif "lora" in name.lower():
            lora_params.append(param)
        else:
            lora_params.append(param)

    print(f"LoRA parameters: {len(lora_params)}")
    print(f"Bias parameters: {len(bias_params)}")

    if config.optimizer_type == "muon":
        lora_optimizer = torch.optim.Muon(
            lora_params,
            lr=config.phase1_learning_rate,
            momentum=config.muon_momentum,
            backend=config.muon_backend,
        )
    else:
        lora_optimizer = torch.optim.AdamW(
            lora_params,
            lr=config.phase1_learning_rate,
            betas=(config.adamw_beta1, config.adamw_beta2),
            weight_decay=config.adamw_weight_decay,
        )

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
    model.train()
    total_loss = 0.0
    num_batches = 0
    progress_bar = tqdm(dataloader, desc=desc)

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        lora_optimizer.step()
        lora_optimizer.zero_grad()

        if bias_optimizer is not None:
            bias_optimizer.step()
            bias_optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

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

        if log_wandb:
            import wandb
            wandb.log({f"{phase_key}/avg_loss": avg_loss, f"{phase_key}/epoch": epoch + 1})

    return losses


def save_checkpoint(model: PreTrainedModel, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
