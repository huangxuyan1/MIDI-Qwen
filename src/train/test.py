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
        track_start_tokens: list[str],     # e.g. ["Track_Start", "Infill_Track"]
        bar_tokens: list[str],             # e.g. ["Bar_None", "Infill_Bar"]
        infill_bar_token: str,             # e.g. "Infill_Bar"
        fillbar_start_token: str,          # e.g. "FillBar_Start"
        bar_mod: int = 64,
        bar_width: int = 512,
        bar_token_increments_before: bool = True,
        debug_print_first_n_batches: int = 0,
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

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
