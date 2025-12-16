#!/usr/bin/env python
import os
import json
import argparse

import torch
from torch.utils.data import DataLoader

from transformers import AutoConfig, AutoModelForCausalLM, set_seed
from miditok import MMM, TokSequence

from ..utils.dataset import MIDIDataset, DataCollatorNoneFilter

from tqdm import tqdm


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


class QwenTimeAwareBarPosCollator:
    """
    Time-aware RoPE position_ids for MMM track-first sequences.

    Markers are specified as TOKEN STRINGS (not ids).
    Works even if BPE merges markers into larger tokens by decoding each tid to strings.

    Important: position_ids advanced ONCE per *BPE token*.
    Decoding is only used for marker detection and state transitions.
    """

    def __init__(
        self,
        base_collator,
        pad_token_id: int,
        mmm,  # MidiTok MMM tokenizer instance (trained)
        track_start_tokens=("Track_Start", "Infill_Track"),
        bar_tokens=("Bar_None", "Infill_Bar"),
        infill_bar_token="Infill_Bar",
        fillbar_start_token="FillBar_Start",
        bar_mod: int = 64,
        bar_width: int = 512,
        bar_token_increments_before: bool = True,
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

        # Cache: bpe_id -> list[str] decoded tokens
        self._decode_cache: dict[int, list[str]] = {}

    def _decode_tid(self, tid: int) -> list[str]:
        cached = self._decode_cache.get(tid)
        if cached is not None:
            return cached
        seq = TokSequence(ids=[tid], are_ids_encoded=True)
        self.mmm.decode_token_ids(seq)
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

            saw_bar_token = False

            # marker detection only
            for tok in decoded:
                if tok in self.track_start_tokens:
                    bar_index = 0

                if tok == self.fillbar_start_token and saved_bar_index_for_fill is not None:
                    bar_index = int(saved_bar_index_for_fill)

                # latch the FIRST infill bar boundary only
                if tok == self.infill_bar_token and saved_bar_index_for_fill is None:
                    saved_bar_index_for_fill = bar_index

                if tok in self.bar_tokens:
                    saw_bar_token = True
                    if self.bar_token_increments_before:
                        bar_index += 1

            ensure_ls(bar_index)
            offset = ls[bar_index]
            max_offset = max(max_offset, offset)
            if offset >= self.bar_width:
                wrap_tokens += 1

            pos[i] = (bar_index % self.bar_mod) * self.bar_width + (offset % self.bar_width)
            ls[bar_index] += 1  # once per BPE token

            if saw_bar_token and (not self.bar_token_increments_before):
                bar_index += 1

        return pos, {"nonpad": nonpad, "max_offset": max_offset, "wrap_tokens": wrap_tokens}

    @torch.no_grad()
    def _compute_position_ids(self, input_ids: torch.Tensor):
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
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "_timeaware_stats": stats,  # optional debug payload
        }


def build_model(qwen_config_dir: str, vocab_size: int, bos_id: int | None, eos_id: int | None, pad_id: int):
    cfg = AutoConfig.from_pretrained(qwen_config_dir)
    cfg.vocab_size = vocab_size
    if bos_id is not None:
        cfg.bos_token_id = bos_id
    if eos_id is not None:
        cfg.eos_token_id = eos_id
    cfg.pad_token_id = pad_id
    return AutoModelForCausalLM.from_config(cfg)


def dump_tokens_and_posids_json(mmm: MMM, input_ids_1d: torch.Tensor, position_ids_1d: torch.Tensor, pad_id: int, out_path: str, limit: int = 200):
    # only nonpad
    nonpad = int((input_ids_1d != pad_id).sum().item())
    n = min(nonpad, limit)

    items = []
    for i in range(n):
        tid = int(input_ids_1d[i].item())
        # decode to token strings (may expand to multiple)
        seq = TokSequence(ids=[tid], are_ids_encoded=True)
        mmm.decode_token_ids(seq)
        toks = list(seq.tokens) if seq.tokens else [f"UNK_{tid}"]
        items.append({
            "i": i,
            "id": tid,
            "tokens": toks,  # list[str]
            "pos_id": int(position_ids_1d[i].item()),
        })

    with open(out_path, "w") as f:
        json.dump({
            "nonpad": nonpad,
            "shown": n,
            "items": items,
        }, f, indent=2)

def scan_max_position_id(
    dataset,
    collator,
    pad_token_id,
    max_samples=1000,
    batch_size=4,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    max_pos = 0
    max_info = None
    seen = 0

    for batch in tqdm(loader, total=max_samples // batch_size + 1):
        pos = batch["position_ids"]          # [B, L]
        inp = batch["input_ids"]

        # ignore pad
        mask = inp != pad_token_id
        cur_max = pos[mask].max().item()

        if cur_max > max_pos:
            max_pos = cur_max
            # store a little context
            idx = (pos == cur_max).nonzero(as_tuple=False)[0]
            b, t = int(idx[0]), int(idx[1])
            max_info = {
                "max_pos": cur_max,
                "batch_index": seen,
                "sample_in_batch": b,
                "token_index": t,
            }

        seen += pos.size(0)
        if seen >= max_samples:
            break

    return max_pos, max_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_disk_path", type=str, required=True, help="Path to load_from_disk dataset (your cached HF dataset)")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--mmm_config", type=str, required=True)
    parser.add_argument("--qwen_config_dir", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--out_json", type=str, default="posids_sample.json")
    args = parser.parse_args()

    set_seed(42)

    from datasets import load_from_disk
    print("Loading dataset from disk...")
    hf_ds = load_from_disk(args.hf_disk_path)
    train_hf = hf_ds[args.train_split]

    print("Initializing MMM tokenizer...")
    mmm = MMM(params=args.mmm_config)

    bos_id = mmm.vocab.get("BOS_None", None)
    eos_id = mmm.vocab.get("EOS_None", None)
    pad_id = mmm.vocab.get("PAD_None", 0)

    print("Building MIDIDataset...")
    ds = MIDIDataset(
        dataset=train_hf,
        tokenizer=mmm,
        max_seq_len=args.max_seq_len,
        ratio_random_tracks_range=(0.4, 1.0),
        data_augmentation_offsets=(6, 2, 0),
        bar_fill_ratio=0.75,
        bar_masking_duration_ratio_range=(0.1, 0.4),
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        force_bar_infilling=True,
        use_attribute_controls=False,
        sample_key_name="input_ids",
        decoder_key_name="decoder_input_ids",
        labels_key_name="labels",
        func_to_get_labels=None,
        ac_random_ratio_range=(0.05, 0.9),
        ac_tracks_random_ratio_range=(0.1, 1.0),
        ac_bars_random_ratio_range=(0.1, 0.7),
    )

    base = DataCollatorNoneFilter(pad_token_id=pad_id, max_length=args.max_seq_len)
    vanilla_collator = QwenDataCollator(base, pad_id)
    timeaware_collator = QwenTimeAwareBarPosCollator(base, pad_id, mmm)

    print("Building model...")
    model = build_model(args.qwen_config_dir, args.vocab_size, bos_id, eos_id, pad_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # create one batch using time-aware collator (so we have position_ids)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=timeaware_collator)
    batch = next(iter(loader))

    # move tensors to device
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    assert "position_ids" in batch, "Time-aware collator did not produce position_ids."

    # Also build a vanilla batch for comparison (same samples)
    # easiest: reuse the same underlying samples
    samples = [ds[i] for i in range(args.batch_size)]
    vanilla = vanilla_collator(samples)
    vanilla = {k: v.to(device) for k, v in vanilla.items()}

    print("Batch shapes:",
          "input_ids", tuple(batch["input_ids"].shape),
          "position_ids", tuple(batch["position_ids"].shape))

    # -----------------------------
    # 1) A/B test: shift position_ids and see logits change
    # -----------------------------
    with torch.no_grad():
        out_a = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
        ).logits

        out_b = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"] + 123,  # big shift
        ).logits

        diff_ab = (out_a - out_b).abs().max().item()

    print(f"[A/B] max |logits diff| (pos_ids vs pos_ids+123): {diff_ab:.6g}")
    if diff_ab < 1e-8:
        print("WARNING: logits did not change -> model likely ignoring supplied position_ids.")

    # -----------------------------
    # 2) With vs without position_ids
    # -----------------------------
    with torch.no_grad():
        out_with = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
        ).logits

        out_without = model(
            input_ids=vanilla["input_ids"],
            attention_mask=vanilla["attention_mask"],
        ).logits

        diff_with_without = (out_with - out_without).abs().max().item()

    print(f"[With/Without] max |logits diff| (timeaware vs vanilla): {diff_with_without:.6g}")
    if diff_with_without < 1e-8:
        print("WARNING: timeaware vs vanilla produced same logits -> position_ids might not be used or batches differ unexpectedly.")

    # -----------------------------
    # 3) Dump sample tokens + pos ids
    # -----------------------------
    out_path = args.out_json
    dump_tokens_and_posids_json(
        mmm=mmm,
        input_ids_1d=batch["input_ids"][1].detach().cpu(),
        position_ids_1d=batch["position_ids"][1].detach().cpu(),
        pad_id=pad_id,
        out_path=out_path,
        limit=2048,
    )
    print(f"Wrote token/pos sample JSON to: {out_path}")

    # Optional: print first 10 entries quickly
    with open(out_path, "r") as f:
        j = json.load(f)
    for row in j["items"][:10]:
        print(row)

    unk_tid = 6657
    print("is_trained:", mmm.is_trained)
    print("model exists:", mmm._model is not None)

    # does HF tokenizer model know this id?
    try:
        tok = mmm._model.id_to_token(unk_tid)
        seq = TokSequence(ids=[unk_tid], are_ids_encoded=True)
        mmm.decode_token_ids(seq)
        print(seq)

        print("hf id_to_token:", repr(tok))
    except Exception as e:
        print("id_to_token failed:", e)

    max_pos, info = scan_max_position_id(
        dataset=ds,
        collator=timeaware_collator,   # your QwenTimeAwareBarPosCollator
        pad_token_id=pad_id,
        max_samples=1000,
        batch_size=4,
    )

    print("MAX position_id:", max_pos)
    print("INFO:", info)
        


if __name__ == "__main__":
    main()
