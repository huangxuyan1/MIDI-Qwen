#!/usr/bin/env python3
"""
Inspect one (or a few) batches from MIDIDataset and dump to disk.

Example:

python inspect_batch.py \
  --dataset_path /fs/scratch/PAS3150/gigamidi_filtered_bars8_notes100 \
  --train_split train \
  --mmm_config /users/PAS3150/alvinh/music_infilling/configs/tokenizer/tokenizer_100k.json \
  --max_seq_len 2048 \
  --batch_size 2 \
  --num_batches 1 \
  --out_dir /users/PAS3150/alvinh/music_infilling/batch_debug \
  --decode_tokens 1 \
  --max_decode_tokens 2048

Notes:
- Uses your MIDIDataset + DataCollatorNoneFilter.
- Dumps JSONL + TXT for human readability.
"""

import os
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import load_from_disk
from miditok import MMM, TokSequence

# Import your dataset + collator
from dataset import MIDIDataset, DataCollatorNoneFilter


# -----------------------------
# Collator wrapper (Trainer-style)
# -----------------------------
class QwenDataCollator:
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


# -----------------------------
# Helpers
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", type=str, required=True,
                   help="Path to load_from_disk dataset (DatasetDict).")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--mmm_config", type=str, required=True)
    p.add_argument("--max_seq_len", type=int, default=2048)

    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_batches", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle", type=int, default=0, help="0/1 to shuffle the loader.")
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--out_dir", type=str, required=True)

    # Decoding controls
    p.add_argument("--decode_tokens", type=int, default=1, help="0/1 decode to token strings.")
    p.add_argument("--max_decode_tokens", type=int, default=600,
                   help="Max tokens to decode per sample for readability.")
    p.add_argument("--dump_labels", type=int, default=0,
                   help="0/1 dump decoded labels too (labels are shifted).")
    p.add_argument("--dump_full_ids", type=int, default=0,
                   help="0/1 dump full input_ids arrays (can be huge).")

    return p.parse_args()


def tensor_stats(name: str, t: torch.Tensor) -> dict:
    return {
        "name": name,
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "min": int(t.min().item()),
        "max": int(t.max().item()),
        "numel": int(t.numel()),
    }


def safe_decode_tokens(mmm: MMM, ids: list[int]):
    """
    Decode token ids into token strings using MMM.
    miditok's decode_token_ids accepts either list[int] or TokSequence depending on version.
    """
    print("Decoding")
    seq = TokSequence(ids=ids, are_ids_encoded=True)
    mmm.decode_token_ids(seq)
    print(seq)
    # exit()
    if hasattr(seq, "tokens") and seq.tokens is not None:
        return seq.tokens
    return None


def write_text_dump(path: Path, records: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write("=" * 100 + "\n")
            f.write(f"batch_idx={rec['batch_idx']} sample_idx={rec['sample_idx']}\n")
            f.write(f"input_ids_shape={rec['input_ids_shape']} pad_token_id={rec['pad_token_id']}\n")
            f.write(f"nonpad_tokens={rec['nonpad_tokens']} loss_tokens={rec['loss_tokens']}\n")
            f.write("\nFirst 80 input_ids:\n")
            f.write(str(rec["input_ids_preview"]) + "\n")
            if rec.get("decoded_tokens") is not None:
                f.write("\nDecoded tokens (preview):\n")
                f.write(" ".join(rec["decoded_tokens_preview"]) + "\n")

            if rec.get("labels_preview") is not None:
                f.write("\nFirst 80 labels (incl -100):\n")
                f.write(str(rec["labels_preview"]) + "\n")
            if rec.get("decoded_labels_preview") is not None:
                f.write("\nDecoded labels (preview):\n")
                f.write(" ".join(rec["decoded_labels_preview"]) + "\n")
            f.write("\n")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset from disk...")
    ds = load_from_disk(args.dataset_path)
    if args.train_split not in ds:
        raise ValueError(f"Split '{args.train_split}' not found. Available: {list(ds.keys())}")
    train_hf = ds[args.train_split]

    print("Initializing MMM tokenizer...")
    mmm = MMM(params=args.mmm_config)

    vocab = mmm.vocab_model
    print(f"  Vocab size: {len(vocab)} tokens")
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as vf:
        json.dump(vocab, vf, indent=2, ensure_ascii=False)

    bos_token_id = mmm.vocab.get("BOS_None", None)
    eos_token_id = mmm.vocab.get("EOS_None", None)
    pad_token_id = mmm.vocab.get("PAD_None", 0)

    print("Building MIDIDataset...")
    train_dataset = MIDIDataset(
        dataset=train_hf,
        tokenizer=mmm,
        max_seq_len=args.max_seq_len,
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

    base_collator = DataCollatorNoneFilter(
        pad_token_id=pad_token_id,
        max_length=args.max_seq_len,
    )
    collator = QwenDataCollator(base_collator, pad_token_id)

    print("Building DataLoader...")
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=bool(args.shuffle),
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    jsonl_path = out_dir / "batch_dump1.jsonl"
    txt_path = out_dir / "batch_dump1.txt"
    meta_path = out_dir / "batch_meta1.json"

    records_for_text = []

    meta = {
        "dataset_path": args.dataset_path,
        "split": args.train_split,
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "max_seq_len": args.max_seq_len,
        "pad_token_id": pad_token_id,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "mmm_config": args.mmm_config,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Writing JSONL to: {jsonl_path}")
    print(f"Writing TXT  to: {txt_path}")
    print(f"Writing META to: {meta_path}")

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.num_batches:
                break

            input_ids = batch["input_ids"]          # (B, L)
            labels = batch["labels"]                # (B, L)
            attention_mask = batch["attention_mask"]

            print(f"\nBatch {batch_idx}:")
            print(" ", tensor_stats("input_ids", input_ids))
            print(" ", tensor_stats("labels", labels))
            print(" ", tensor_stats("attention_mask", attention_mask))

            B, L = input_ids.shape

            for b in range(B):
                ids = input_ids[b].tolist()
                lab = labels[b].tolist()

                # determine non-pad region for readability
                # (note: labels may be -100 even where input is not pad)
                nonpad = sum(1 for t in ids if t != pad_token_id)
                loss_tokens = sum(1 for v in lab[:nonpad] if v != -100)

                preview_len = min(args.max_decode_tokens, nonpad)
                ids_preview = ids[:preview_len]
                lab_preview = lab[:preview_len]

                decoded_tokens = None
                decoded_tokens_preview = None
                if args.decode_tokens:
                    decoded_tokens_preview = safe_decode_tokens(mmm, ids_preview)

                decoded_labels_preview = None
                if args.dump_labels and args.decode_tokens:
                    # decode labels where not -100; replace -100 with PAD for decoding safety
                    lab_ids = [(pad_token_id if v == -100 else v) for v in lab_preview]
                    decoded_labels_preview = safe_decode_tokens(mmm, lab_ids)

                record = {
                    "batch_idx": batch_idx,
                    "sample_idx": b,
                    "pad_token_id": pad_token_id,
                    "input_ids_shape": [int(B), int(L)],
                    "nonpad_tokens": int(nonpad),
                    "loss_tokens": int(loss_tokens),
                    "input_ids_preview": ids_preview[:80],
                    "labels_preview": (lab_preview[:80] if args.dump_labels else None),
                    "decoded_tokens_preview": decoded_tokens_preview,
                    "decoded_labels_preview": (decoded_labels_preview if args.dump_labels else None),
                }

                if args.dump_full_ids:
                    # Warning: huge. Use only when needed.
                    record["input_ids_full"] = ids
                    record["labels_full"] = lab

                jf.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_for_text.append(record)

    write_text_dump(txt_path, records_for_text)

    print("\nDone.")
    print(f"Open: {txt_path}")
    print(f"Or parse JSONL: {jsonl_path}")


if __name__ == "__main__":
    main()
