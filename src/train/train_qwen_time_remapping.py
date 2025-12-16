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

from datasets import load_dataset, load_from_disk, DatasetDict
from miditok import MMM
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ..utils.dataset import MIDIDataset, DataCollatorNoneFilter
import dotenv


# ---------------------------------------------------------------------------
# Data collator wrapper to match Qwen / Trainer API
# ---------------------------------------------------------------------------

import torch
from miditok import TokSequence


class QwenTimeAwareBarPosCollator:
    """
    Time-aware RoPE position_ids for MMM track-first sequences.

    Markers are specified as TOKEN STRINGS (not ids).
    Works even if BPE merges markers into larger tokens by decoding each tid to strings.
    """

    def __init__(
        self,
        base_collator,
        pad_token_id: int,
        mmm,  # MidiTok MMM tokenizer instance (trained)
        track_start_tokens: list[str] = ["Track_Start", "Infill_Track"],    # e.g. ["Track_Start", "Infill_Track"]
        bar_tokens: list[str] = ["Bar_None", "Infill_Bar"],                 # e.g. ["Bar_None", "Infill_Bar"]
        infill_bar_token: str = "Infill_Bar",                               # e.g. "Infill_Bar"
        fillbar_start_token: str = "FillBar_Start",                         # e.g. "FillBar_Start"
        bar_mod: int = 64,
        bar_width: int = 512,
        bar_token_increments_before: bool = True,
        debug_print_first_n_batches: int = 1,
        max_position_embeddings: int | None = None, 
        clamp_log_path: str | None = None
    ):
        self.base_collator = base_collator
        self.pad_token_id = pad_token_id
        self.mmm = mmm

        self.track_start_tokens = set(track_start_tokens)
        self.bar_tokens = set(bar_tokens)
        self.infill_bar_token = infill_bar_token
        self.fillbar_start_token = fillbar_start_token

        self.bar_mod = bar_mod
        self.bar_width = bar_width
        self.bar_token_increments_before = bar_token_increments_before

        self.debug_print_first_n_batches = debug_print_first_n_batches
        self._seen_batches = 0

        # Cache: bpe_id -> list[str] decoded tokens
        self._decode_cache: dict[int, list[str]] = {}
        self.max_position_embeddings = max_position_embeddings
        self.clamp_log_path = clamp_log_path
        self._clamp_logged = False

    def _decode_tid(self, tid: int) -> list[str]:
        """Decode a (possibly BPE) id into underlying token strings. Cached."""
        cached = self._decode_cache.get(tid)
        if cached is not None:
            return cached

        seq = TokSequence(ids=[tid], are_ids_encoded=True)
        self.mmm.decode_token_ids(seq)  # populates seq.tokens
        toks = list(seq.tokens) if seq.tokens else [f"UNK_{tid}"]

        self._decode_cache[tid] = toks
        return toks

    @torch.no_grad()
    def _compute_position_ids_one(self, ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        L = ids.numel()
        pos = torch.zeros((L,), dtype=torch.long, device=ids.device)

        nonpad = int((ids != self.pad_token_id).sum().item())
        if nonpad == 0:
            return pos, {"nonpad": 0, "max_offset": 0, "wrap_tokens": 0}

        bar_index = 0
        ls = [0]  # persistent across track starts
        saved_bar_index_for_fill = None

        max_offset = 0
        wrap_tokens = 0

        def ensure_ls(i: int):
            while len(ls) <= i:
                ls.append(0)

        for i in range(nonpad):
            tid = int(ids[i].item())
            decoded = self._decode_tid(tid)

            # --- detection / state transitions only ---
            saw_bar_token = False

            for tok in decoded:
                if tok in self.track_start_tokens:
                    bar_index = 0

                if tok == self.fillbar_start_token and saved_bar_index_for_fill is not None:
                    bar_index = int(saved_bar_index_for_fill)

                if tok == self.infill_bar_token and saved_bar_index_for_fill is None:
                    # store current bar index BEFORE any bar increment (your spec)
                    saved_bar_index_for_fill = bar_index

                if tok in self.bar_tokens:
                    saw_bar_token = True
                    if self.bar_token_increments_before:
                        bar_index += 1

            # --- assign ONE position for this BPE token ---
            ensure_ls(bar_index)
            offset = ls[bar_index]
            max_offset = max(max_offset, offset)
            if offset >= self.bar_width:
                wrap_tokens += 1

            pos[i] = (bar_index % self.bar_mod) * self.bar_width + (offset % self.bar_width)

            # advance ONCE per BPE token
            ls[bar_index] += 1

            # increment-after mode
            if saw_bar_token and (not self.bar_token_increments_before):
                bar_index += 1

        return pos, {"nonpad": nonpad, "max_offset": max_offset, "wrap_tokens": wrap_tokens}

    @torch.no_grad()
    def _compute_position_ids(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[dict]]:
        B, L = input_ids.shape
        out = torch.zeros((B, L), dtype=torch.long, device=input_ids.device)
        stats = []
        for b in range(B):
            out[b], st = self._compute_position_ids_one(input_ids[b])
            stats.append(st)
        return out, stats

    def __call__(self, batch):
        input_ids, labels = self.base_collator(batch)
        attention_mask = (input_ids != self.pad_token_id).long()
        position_ids, stats = self._compute_position_ids(input_ids)

        self._seen_batches += 1
        if self.debug_print_first_n_batches and self._seen_batches <= self.debug_print_first_n_batches:
            for si, st in enumerate(stats):
                print(
                    f"[TimeAwarePE] batch={self._seen_batches} sample={si} "
                    f"nonpad={st['nonpad']} max_offset={st['max_offset']} wrap_tokens={st['wrap_tokens']}"
                )

        if self.max_position_embeddings is not None:
            max_before = int(position_ids.max().item())
            limit = self.max_position_embeddings - 1

            if max_before > limit:
                position_ids = position_ids.clamp_max(limit)

                # log ONCE per worker
                if (not self._clamp_logged) and self.clamp_log_path is not None:
                    try:
                        with open(self.clamp_log_path, "a") as f:
                            f.write(
                                f"[TimeAwarePE][CLAMP] "
                                f"max_before={max_before} limit={limit}\n"
                            )
                    except Exception:
                        pass  # never crash training for logging

                    self._clamp_logged = True

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
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

    vocab_size = 16000
    config.vocab_size = vocab_size

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
    num_workers = train_cfg.get("num_workers", 4)

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
    # hf_ds = load_dataset(
    #     args.dataset_name,
    #     args.dataset_config,
    #     token=os.getenv("HF_TOKEN"),
    # )
    hf_ds = load_from_disk("/fs/scratch/PAS3150/gigamidi_filtered_bars8_notes100") 
    train_hf = hf_ds[args.train_split]
    eval_hf = hf_ds[args.eval_split] if args.eval_split is not None else None

    if eval_hf is not None:
        max_eval = 1000  # Limit eval to 1000 samples
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
        force_bar_infilling=True,
        use_attribute_controls=False,
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
            force_bar_infilling=True,
            use_attribute_controls=False,
        )

    # 5) Collator: your DataCollatorNoneFilter + Qwen wrapper
    base_collator = DataCollatorNoneFilter(
        pad_token_id=pad_token_id,
        max_length=max_seq_len,
    )

    clamp_log = os.path.join(args.output_dir, "position_id_clamp.log")
    data_collator = QwenTimeAwareBarPosCollator(
        base_collator, 
        pad_token_id, 
        mmm_tokenizer, 
        max_position_embeddings=model.config.max_position_embeddings,
        clamp_log_path=clamp_log
    )

    # Sanity check: single batch -----------------------------------------------------------------
    print("model.config.max_position_embeddings =", model.config.max_position_embeddings)
    print("model.config.rope_scaling =", getattr(model.config, "rope_scaling", None))
    print("model.config.original_max_position_embeddings =", getattr(model.config, "original_max_position_embeddings", None))

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

        dataloader_num_workers=num_workers, 
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=4,
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
