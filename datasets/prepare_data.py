from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_tokenize(dataset_name, config_name, tokenizer_name, split="test", max_samples=100):
    dataset = load_dataset(dataset_name, config_name, split=split)
    dataset = dataset.select(range(max_samples))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def tokenize(example):
        return tokenizer(example["text"], return_attention_mask=False, truncation=True)

    return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
