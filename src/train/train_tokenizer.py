# scripts/train/train_tokenizer.py

from pathlib import Path
from miditok import MMM, TokenizerConfig
from datasets import load_dataset
import dotenv
import os
import json
import random

random.seed(42)

dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def build_tokenizer() -> MMM:

    base_config_path = Path("/users/PAS3150/alvinh/music_infilling/data/meta/tokenizer_midi_rwkv.json")
    with open(base_config_path, "r") as f:
        base_cfg = json.load(f)['config']

    # del base_cfg['pitch_range']
    del base_cfg['beat_res']
    del base_cfg['beat_res_rest']
    # del base_cfg['use_note_duration_programs']
    del base_cfg['time_signature_range']

    cfg = TokenizerConfig.from_dict(base_cfg)

    tokenizer = MMM(tokenizer_config=cfg)
    return tokenizer


def main():
    # 1) collect MIDI files
    ds = load_dataset("Metacreation/GigaMIDI", name="v2.0.0", split="train", token=HF_TOKEN)
    
    # 2) build tokenizer
    tokenizer = build_tokenizer()

    # 3) train BPE on these files
    tokenizer.train(
        vocab_size=16000,      # or whatever MIDI-RWKV used / you want
        files_paths=random.sample(list(Path('/fs/scratch/PAS3150/filtered_gigamidi').glob("*.mid")), 100000)  # choose 100k random files for training
    )

    # 4) save tokenizer into data/meta/tokenizer/
    out_dir = Path("/users/PAS3150/alvinh/music_infilling/data/meta")
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_dir)
    print(f"Saved tokenizer to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
