# src/train/pretrain_midi.py

import os
import argparse
import yaml

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from src.data_loader.midi_packed_dataset import NpzPackedDataset, CausalLMDataCollator


def load_config(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # handle simple include mechanism
    if "include" in cfg:
        merged = {}
        for inc in cfg["include"]:
            inc_cfg = load_config(inc)
            merged.update(inc_cfg)
        # top-level overrides included
        for k, v in cfg.items():
            if k != "include":
                merged[k] = v
        return merged
    return cfg


def get_tokenizer(model_cfg):
    if model_cfg.get("use_custom_tokenizer", False):
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=model_cfg["tokenizer_json"]
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_name"])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train/pretrain_midi_stage1.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    seq_len = model_cfg["seq_len"]

    # -------------------------
    # Datasets
    # -------------------------
    train_dataset = NpzPackedDataset(
        shard_glob=train_cfg["train_glob"],
        seq_len=seq_len,
        shuffle_shards=train_cfg.get("shuffle_shards", False),
    )

    eval_dataset = None
    eval_glob = train_cfg.get("eval_glob", "")
    if eval_glob:
        eval_dataset = NpzPackedDataset(
            shard_glob=eval_glob,
            seq_len=seq_len,
            shuffle_shards=False,
        )

    # -------------------------
    # Model & tokenizer
    # -------------------------
    tokenizer = get_tokenizer(model_cfg)

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_name"],
        torch_dtype=torch.bfloat16
        if model_cfg.get("dtype", "bfloat16") == "bfloat16" and torch.cuda.is_available()
        else torch.float32,
        device_map="auto",
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    # -------------------------
    # TrainingArguments from config
    # -------------------------
    output_dir = train_cfg["output_dir"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        eval_steps=train_cfg["eval_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        evaluation_strategy=train_cfg["evaluation_strategy"]
        if eval_dataset is not None
        else "no",
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        bf16=(model_cfg.get("dtype", "bfloat16") == "bfloat16" and torch.cuda.is_available()),
        tf32=True,
        report_to="none",  # or "wandb"/"tensorboard"
    )

    data_collator = CausalLMDataCollator()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
