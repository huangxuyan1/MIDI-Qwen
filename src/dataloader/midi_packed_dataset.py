# src/data_loader/midi_packed_dataset.py

import glob
from typing import List, Dict, Iterator, Any

import numpy as np
import torch
from torch.utils.data import IterableDataset
from dataclasses import dataclass


class NpzPackedDataset(IterableDataset):
    """
    Streams .npz shards, reads 1D `input_ids` arrays and packs them into
    fixed-length sequences of token IDs.

    Each shard is expected to contain:
        input_ids: 1D array of ints (flattened).
    """

    def __init__(self, shard_glob: str, seq_len: int, shuffle_shards: bool = False):
        super().__init__()
        self.shard_glob = shard_glob
        self.seq_len = seq_len
        self.shuffle_shards = shuffle_shards

    def _shard_paths(self) -> List[str]:
        paths = sorted(glob.glob(self.shard_glob))
        if not paths:
            raise ValueError(f"No shards matched pattern: {self.shard_glob}")
        if self.shuffle_shards:
            rng = np.random.default_rng()
            rng.shuffle(paths)
        return paths

    def _token_iterator(self) -> Iterator[int]:
        for path in self._shard_paths():
            arr = np.load(path)["input_ids"]
            # ensure 1D
            arr = arr.reshape(-1)
            for tid in arr:
                yield int(tid)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        buffer: List[int] = []
        for tid in self._token_iterator():
            buffer.append(tid)
            if len(buffer) >= self.seq_len:
                chunk = buffer[: self.seq_len]
                buffer = buffer[self.seq_len :]
                yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}

        # drop tail; you can pad if you really want to use every token


@dataclass
class CausalLMDataCollator:
    """
    Fixed-length sequences; labels = input_ids, attention_mask = 1s.
    No padding because dataset already emits fixed-length tensors.
    """

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
