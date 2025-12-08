import os
from datasets import load_dataset
from io import BytesIO
import miditoolkit
import math
import dotenv

dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

OUT_DIR = "/fs/scratch/PAS3150/filtered_gigamidi"
os.makedirs(OUT_DIR, exist_ok=True)

# number of worker processes
NUM_PROC = int(os.getenv("NUM_WORKERS", "8"))  # tweak as you like


# --- Helpers ---
def count_notes(midi):
    return sum(len(inst.notes) for inst in midi.instruments)


def estimate_bars(midi):
    """
    Estimate number of bars exactly using time_signature_changes.
    Handles multiple time signatures correctly.
    """

    tpq = midi.ticks_per_beat
    ts_changes = midi.time_signature_changes

    # If no time signatures, assume 4/4 over whole song
    if not ts_changes:
        beats_per_bar = 4
        last_tick = midi.max_tick
        total_beats = last_tick / tpq
        return int(total_beats / beats_per_bar)

    # Sort time signature changes by their start tick
    ts_changes = sorted(ts_changes, key=lambda ts: ts.time)

    # Add an artificial end marker at the end of the file
    # This lets us compute the duration of the last signature segment
    segments = []
    for i, ts in enumerate(ts_changes):
        start = ts.time
        if i + 1 < len(ts_changes):
            end = ts_changes[i + 1].time
        else:
            end = midi.max_tick
        segments.append((ts, start, end))

    total_bars = 0

    for ts, start, end in segments:
        seg_ticks = max(0, end - start)
        seg_beats = seg_ticks / tpq

        # beats per bar = numerator * (4 / denominator)
        beats_per_bar = ts.numerator * (4 / ts.denominator)

        if beats_per_bar > 0:
            bars_in_segment = seg_beats / beats_per_bar
        else:
            bars_in_segment = 0

        total_bars += bars_in_segment

    return int(total_bars)


# --- Per-example worker function ---
def filter_and_save(example, idx):
    """
    This runs in multiple processes via Dataset.map.
    Side effect: writes .mid file to OUT_DIR if it passes the filter.
    """
    midi_bytes = example["music"]
    md5 = example["md5"]

    out_path = os.path.join(OUT_DIR, f"gm_{idx}_{md5}.mid")

    # --- RECOVERY / RESUME LOGIC ---
    # If we've already written this file before a crash, skip re-processing.
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        # We assume it was accepted before, so mark kept=True without recomputing.
        return {"kept": True, "num_bars": -1, "num_notes": -1}

    try:
        midi = miditoolkit.MidiFile(file=BytesIO(midi_bytes))
    except Exception:
        # bad midi, skip
        return {"kept": False, "num_bars": 0, "num_notes": 0}

    notes = count_notes(midi)
    bars = estimate_bars(midi)

    if bars >= 8 and notes >= 100:
        # unique-ish filename; idx is shard-local but good enough when combined with md5
        with open(out_path, "wb") as f:
            f.write(midi_bytes)
        kept = True
    else:
        kept = False

    # returning something small is nice for logging/stats (optional)
    return {"kept": kept, "num_bars": bars, "num_notes": notes}


def main():
    # Load dataset once in the main process
    ds = load_dataset(
        "Metacreation/GigaMIDI",
        name="v2.0.0",
        split="train",
        token=HF_TOKEN,
    )

    # This will spawn NUM_PROC workers and run filter_and_save on shards
    result = ds.map(
        filter_and_save,
        with_indices=True,
        num_proc=NUM_PROC,
        desc="Filtering and saving MIDIs",
    )

    # Count how many were kept (just for info)
    kept_total = sum(result["kept"])
    print("DONE.")
    print("Total kept MIDIs:", kept_total)
    print("Saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
