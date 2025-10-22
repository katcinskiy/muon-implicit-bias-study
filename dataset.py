"""
Dataset creation utilities for the LoRA orthogonalization experiment.
"""

from datasets import Dataset
from transformers import PreTrainedTokenizer


def create_fact_dataset(
    fact: str, num_samples: int, tokenizer: PreTrainedTokenizer, max_length: int = 128
) -> Dataset:
    """
    Create a dataset with repeated fact samples.

    Args:
        fact: The fact string to repeat (e.g., "The capital of France is Paris.")
        num_samples: Number of samples to create
        tokenizer: Tokenizer to use for encoding
        max_length: Maximum sequence length

    Returns:
        Dataset with tokenized samples
    """
    # Create repeated samples
    texts = [fact] * num_samples

    # Tokenize
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    # Create dataset
    dataset = Dataset.from_dict(
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": encodings["input_ids"].clone(),  # For causal LM, labels = input_ids
        }
    )

    return dataset


def create_phase1_dataset(
    config, tokenizer: PreTrainedTokenizer
) -> Dataset:
    """Create Phase 1 dataset (original fact)."""
    return create_fact_dataset(
        fact=config.original_fact,
        num_samples=config.phase1_samples,
        tokenizer=tokenizer,
    )


def create_phase2_dataset(
    config, tokenizer: PreTrainedTokenizer
) -> Dataset:
    """Create Phase 2 dataset (conflicting fact)."""
    return create_fact_dataset(
        fact=config.conflicting_fact,
        num_samples=config.phase2_samples,
        tokenizer=tokenizer,
    )
