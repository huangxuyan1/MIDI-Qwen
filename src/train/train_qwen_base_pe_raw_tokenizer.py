#!/usr/bin/env python
"""
Train a Qwen3 0.6B-style Causal LM from scratch on MMM tokens using MIDIDataset.

Assumptions:
- src/dataset.py defines MIDIDataset and DataCollatorNoneFilter.
- You have a local Qwen3 config directory, e.g. configs/qwen3_0.6b_base/config.json
- You have a MMM tokenizer config JSON, e.g. configs/mmm_config.json
- You have a training hyperparameter JSON, e.g. configs/train_stage1.json

Example train_stage1.json:

{
  "max_seq_len": 2048,
  "per_device_train_batch_size": 4,
  "per_device_eval_batch_size": 4,
  "gradient_accumulation_steps": 4,
  "learning_rate": 0.0002,
  "weight_decay": 0.01,
  "warmup_ratio": 0.03,
  "num_train_epochs": 1,
  "max_train_steps": null,
  "logging_steps": 50,
  "save_steps": 1000,
  "eval_steps": 1000,
  "save_total_limit": 3,
  "bf16": true,
  "fp16": false
}

Usage (OSC A100 example):

srun python train_qwen3_mmm.py \
  --dataset_name Metacreation/GigaMIDI \
  --dataset_config v2.0.0 \
  --mmm_config ./configs/mmm_config.json \
  --qwen_config_dir ./configs/qwen3_0.6b_base \
  --train_config ./configs/train_stage1.json \
  --output_dir ./outputs/qwen3_06b_mmm_stage1 \
  --wandb_project midi_qwen \
  --wandb_run_name qwen3_stage1_gigamidi
"""

import os
import json
import argparse

from datasets import load_dataset
from miditok import MMM
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)

from utils.dataset_raw_nohf import MIDIDataset, DataCollatorNoneFilter
import dotenv


# ---------------------------------------------------------------------------
# Data collator wrapper to match Qwen / Trainer API
# ---------------------------------------------------------------------------

class QwenDataCollator:
    """
    Wraps DataCollatorNoneFilter (which returns (input_ids, labels)) and converts
    it into a dict suitable for HF Trainer / Qwen:

        {
            "input_ids": ...,
            "labels": ...,
            "attention_mask": ...
        }
    """

    def __init__(self, base_collator: DataCollatorNoneFilter, pad_token_id: int):
        self.base_collator = base_collator
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids, labels = self.base_collator(batch)
        attention_mask = (input_ids != self.pad_token_id).long()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Hugging Face dataset name, e.g. Metacreation/GigaMIDI",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="HF dataset config/subset, e.g. v2.0.0",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Name of training split.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="validation",
        help="Optional eval split; if None, no eval.",
    )

    # Configs
    parser.add_argument(
        "--mmm_config",
        type=str,
        required=True,
        help="Path to MMM tokenizer config JSON.",
    )
    parser.add_argument(
        "--qwen_config_dir",
        type=str,
        required=True,
        help="Path to local Qwen config directory (containing config.json).",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        required=True,
        help="Path to training hyperparameter JSON.",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save checkpoints and logs.",
    )

    # Wandb
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Weights & Biases project name. If None, wandb is disabled.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional wandb run name.",
    )

    return parser.parse_args()


def load_train_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg


def build_qwen3_mmm_model(qwen_config_dir: str, mmm_tokenizer: MMM):
    """
    Load Qwen3 config from a local directory, override vocab_size and token IDs
    to match MMM tokenizer, and create a fresh Qwen3ForCausalLM model.
    """
    config = AutoConfig.from_pretrained(qwen_config_dir)

    vocab_size = mmm_tokenizer.vocab_size
    config.vocab_size = vocab_size
    print(f"Setting model vocab_size to {vocab_size} from MMM tokenizer.")

    bos_token_id = mmm_tokenizer.vocab.get("BOS_None")
    eos_token_id = mmm_tokenizer.vocab.get("EOS_None")
    pad_token_id = mmm_tokenizer.vocab.get("PAD_None", 0)

    # Override BOS/EOS/PAD to MMM specials
    if bos_token_id is not None:
        config.bos_token_id = bos_token_id
    if eos_token_id is not None:
        config.eos_token_id = eos_token_id
    config.pad_token_id = pad_token_id

    model = AutoModelForCausalLM.from_config(config)
    return model, pad_token_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dotenv.load_dotenv()  # Load .env for HF token if exists

    args = parse_args()
    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load training hyperparameters
    train_cfg = load_train_config(args.train_config)

    max_seq_len = train_cfg.get("max_seq_len", 2048)
    per_device_train_batch_size = train_cfg.get("per_device_train_batch_size", 4)
    per_device_eval_batch_size = train_cfg.get("per_device_eval_batch_size", 4)
    gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 1)
    learning_rate = train_cfg.get("learning_rate", 2e-4)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    warmup_ratio = train_cfg.get("warmup_ratio", 0.03)
    num_train_epochs = train_cfg.get("num_train_epochs", 1.0)
    max_train_steps = train_cfg.get("max_train_steps", None)
    logging_steps = train_cfg.get("logging_steps", 50)
    save_steps = train_cfg.get("save_steps", 1000)
    eval_steps = train_cfg.get("eval_steps", 1000)
    save_total_limit = train_cfg.get("save_total_limit", 3)
    bf16 = train_cfg.get("bf16", False)
    fp16 = train_cfg.get("fp16", False)

    # Configure wandb
    if args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_run_name is not None:
            os.environ["WANDB_NAME"] = args.wandb_run_name
        report_to = ["wandb"]
    else:
        # You can add "tensorboard" here if desired
        report_to = ["none"]

    # 1) Load HF dataset
    print("Loading dataset...")
    hf_ds = load_dataset(
        args.dataset_name,
        args.dataset_config,
        token=os.getenv("HF_TOKEN"),
    )
    train_hf = hf_ds[args.train_split]
    eval_hf = hf_ds[args.eval_split] if args.eval_split is not None else None

    if eval_hf is not None:
        max_eval = 2000  # or 1000
        eval_hf = eval_hf.shuffle(seed=42).select(range(min(max_eval, len(eval_hf))))

    # 2) MMM tokenizer
    print("Initializing MMM tokenizer...")
    mmm_tokenizer = MMM(params=args.mmm_config)
    bos_token_id = mmm_tokenizer.vocab.get("BOS_None", None)
    eos_token_id = mmm_tokenizer.vocab.get("EOS_None", None)

    # 3) Build Qwen3 0.6B-style model using local config
    print("Building Qwen3-0.6B-style model from local config...")
    model, pad_token_id = build_qwen3_mmm_model(args.qwen_config_dir, mmm_tokenizer)

    # 4) Build MIDIDataset with midi-rwkv paper defaults you provided
    print("Building MIDIDataset (train)...")
    train_dataset = MIDIDataset(
        dataset=train_hf,
        tokenizer=mmm_tokenizer,
        max_seq_len=max_seq_len,
        ratio_random_tracks_range=(0.4, 1.0),
        data_augmentation_offsets=(6, 2, 0),
        bar_fill_ratio=0.75,
        bar_masking_duration_ratio_range=(0.1, 0.4),
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        ac_random_ratio_range=(0.05, 0.9),
        ac_tracks_random_ratio_range=(0.1, 1.0),
        ac_bars_random_ratio_range=(0.1, 0.7),
        func_to_get_labels=None,
        sample_key_name="input_ids",
        decoder_key_name="decoder_input_ids",
        labels_key_name="labels",
    )

    eval_dataset = None
    if eval_hf is not None:
        print("Building MIDIDataset (eval)...")
        eval_dataset = MIDIDataset(
            dataset=eval_hf,
            tokenizer=mmm_tokenizer,
            max_seq_len=max_seq_len,
            ratio_random_tracks_range=(0.4, 1.0),
            data_augmentation_offsets=(6, 2, 0),
            bar_fill_ratio=0.75,
            bar_masking_duration_ratio_range=(0.1, 0.4),
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            ac_random_ratio_range=(0.05, 0.9),
            ac_tracks_random_ratio_range=(0.1, 1.0),
            ac_bars_random_ratio_range=(0.1, 0.7),
            func_to_get_labels=None,
            sample_key_name="input_ids",
            decoder_key_name="decoder_input_ids",
            labels_key_name="labels",
        )

    # 5) Collator: your DataCollatorNoneFilter + Qwen wrapper
    base_collator = DataCollatorNoneFilter(
        pad_token_id=pad_token_id,
        max_length=max_seq_len,
    )
    data_collator = QwenDataCollator(base_collator, pad_token_id)

    # Sanity check: single batch -----------------------------------------------------------------
    from torch.utils.data import DataLoader
    loader = DataLoader(train_dataset, batch_size=2, shuffle=False, collate_fn=data_collator)
    batch = next(iter(loader))

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    print("input_ids shape:", input_ids.shape)
    print("labels shape:", labels.shape)
    print("input_ids min/max:", input_ids.min().item(), input_ids.max().item())
    valid_labels = labels[labels != -100]
    if valid_labels.numel() > 0:
        print("labels min/max (excluding -100):",
              valid_labels.min().item(), valid_labels.max().item())
    else:
        print("No valid labels (all -100?)")

    vocab_size = model.config.vocab_size
    print("model vocab_size:", vocab_size)

    # Hard assertions to catch any bad token
    assert input_ids.min().item() >= 0, f"Found negative input_ids!, min={input_ids.min().item()}"
    assert input_ids.max().item() < vocab_size, f"Found input_ids >= vocab_size!, max={input_ids.max().item()}"

    if valid_labels.numel() > 0:
        assert valid_labels.min().item() >= 0, "Found negative label (other than -100)!"
        assert valid_labels.max().item() < vocab_size, "Found label >= vocab_size!"

    print("Single batch looks OK.")

    # ---------------------------------------------------------------------------

    # 6) TrainingArguments (from train_config)
    effective_max_steps = max_train_steps if max_train_steps is not None else -1

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        num_train_epochs=num_train_epochs,
        max_steps=effective_max_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        eval_strategy="steps" if eval_dataset is not None else "no",
        logging_strategy="steps",
        bf16=bf16,
        fp16=fp16,
        report_to=report_to,
        remove_unused_columns=False,
        run_name=args.wandb_run_name,
    )

    # 7) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # 8) Train
    print("Starting training...")
    trainer.train()
    print("Training finished. Saving model...")
    trainer.save_model()
    print("Done.")


if __name__ == "__main__":
    main()
