from dataclasses import dataclass
from typing import Literal


@dataclass
class ExperimentConfig:
    model_name: str = "openai-community/gpt2"

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: list[str] = None

    phase1_samples: int = 1000
    phase1_epochs: int = 3
    phase1_batch_size: int = 8
    phase1_learning_rate: float = 3e-4

    phase2_samples: int = 10
    phase2_epochs: int = 3
    phase2_batch_size: int = 8
    phase2_learning_rate: float = 3e-4

    optimizer_type: Literal["adamw", "muon"] = "muon"
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999
    adamw_weight_decay: float = 0.01
    muon_momentum: float = 0.95
    muon_backend: str = "newtonschulz5"

    original_fact: str = "The capital of France is Paris."
    conflicting_fact: str = "The capital of France is Lyon."
    eval_prompt: str = "The capital of France is"

    output_dir: str = "./results"
    seed: int = 42

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["c_attn"]
