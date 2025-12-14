import io
from typing import Dict, Any, List

import miditoolkit
from datasets import load_dataset, DatasetDict

import os
import dotenv


def _count_bars_miditoolkit(midi: miditoolkit.MidiFile) -> float:
    """
    Returns bar count as a float (fractional allowed) by integrating over
    time-signature segments up to the last note end tick.
    """
    ppq = midi.ticks_per_beat

    # max_tick across all notes (end time)
    max_tick = 0
    for inst in midi.instruments:
        for n in inst.notes:
            if n.end > max_tick:
                max_tick = n.end
    if max_tick <= 0:
        return 0.0

    # global time signatures
    tss = list(midi.time_signature_changes or [])
    tss.sort(key=lambda ts: ts.time)

    # default TS at tick 0 if missing
    if not tss or tss[0].time != 0:
        tss.insert(0, miditoolkit.TimeSignature(4, 4, 0))

    bars = 0.0
    for i, ts in enumerate(tss):
        seg_start = ts.time
        if seg_start >= max_tick:
            break

        seg_end = max_tick
        if i + 1 < len(tss):
            seg_end = min(seg_end, tss[i + 1].time)

        if seg_end <= seg_start:
            continue

        ticks_per_bar = ppq * ts.numerator * (4.0 / ts.denominator)
        bars += (seg_end - seg_start) / ticks_per_bar

    return bars


def bars_from_midi_bytes(midi_bytes: bytes) -> float:
    """
    Parse a MIDI from bytes and return integrated bar count.
    Returns 0.0 if parsing fails.
    """
    try:
        bio = io.BytesIO(midi_bytes)
        midi = miditoolkit.MidiFile(file=bio)
        return _count_bars_miditoolkit(midi)
    except Exception:
        return 0.0


def compute_num_bars_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[float]]:
    """
    HF datasets batched map: expects batch["music"] to be a list of bytes-like objects.
    """
    out = []
    for x in batch["music"]:
        # Most HF datasets store bytes as Python bytes already.
        # If it's memoryview/bytearray, bytes(x) handles it.
        try:
            b = x if isinstance(x, (bytes,)) else bytes(x)
        except Exception:
            out.append(0.0)
            continue

        out.append(bars_from_midi_bytes(b))
    return {"num_bars": out}


def main():
    # 1) Load dataset (edit these to match your setup)
    # Example patterns:
    # ds = load_dataset("your_org/gigamidi", split="train")
    # ds_dict = load_dataset("your_org/gigamidi")  # DatasetDict with train/valid/test
    dotenv.load_dotenv()
    ds_dict: DatasetDict = load_dataset("Metacreation/GigaMIDI", "v2.0.0", token=os.getenv("HF_TOKEN"))  # DatasetDict with train/valid/test

    min_total_notes = 100
    min_bars = 8.0
    num_proc = 8  # adjust to your CPU; set 1 if debugging

    def cheap_filter(ex: Dict[str, Any]) -> bool:
        return int(ex["total_notes"]) >= min_total_notes

    filtered_splits = {}
    for split_name, ds in ds_dict.items():
        # 2) Cheap filter first
        ds = ds.filter(cheap_filter, num_proc=num_proc)

        # 3) Compute num_bars
        ds = ds.map(
            compute_num_bars_batch,
            batched=True,
            batch_size=64,
            num_proc=num_proc,
            desc=f"Computing num_bars ({split_name})",
        )

        # 4) Bar filter
        ds = ds.filter(lambda ex: float(ex["num_bars"]) >= min_bars, num_proc=num_proc)

        filtered_splits[split_name] = ds

    out_ds = DatasetDict(filtered_splits)

    # 5) Save as an HF dataset on disk
    out_path = "/fs/scratch/PAS3150/gigamidi_filtered_bars8_notes100"
    out_ds.save_to_disk(out_path)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
