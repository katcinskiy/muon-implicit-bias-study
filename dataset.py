from datasets import Dataset
from transformers import PreTrainedTokenizer


def create_fact_dataset(
    fact: str, num_samples: int, tokenizer: PreTrainedTokenizer, max_length: int = 128
) -> Dataset:
    texts = [fact] * num_samples
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    dataset = Dataset.from_dict(
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": encodings["input_ids"].clone(),
        }
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


def create_phase1_dataset(config, tokenizer: PreTrainedTokenizer) -> Dataset:
    return create_fact_dataset(
        fact=config.original_fact,
        num_samples=config.phase1_samples,
        tokenizer=tokenizer,
    )


def create_phase2_dataset(config, tokenizer: PreTrainedTokenizer) -> Dataset:
    return create_fact_dataset(
        fact=config.conflicting_fact,
        num_samples=config.phase2_samples,
        tokenizer=tokenizer,
    )
