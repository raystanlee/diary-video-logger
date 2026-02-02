from __future__ import annotations

from typing import Any
import torch
from torch.utils.data import Dataset


class CharadesVideoDataset(Dataset):
    """
    Holds (video_path, multi-hot label) pairs for a *subset* of Charades actions.

    - Charades stores labels as list[int] in the global label space (0..156).
    - `label2id` should map global_label_id(int) -> subset_position(int, 0..num_labels-1).

    Decoding is NOT done here. Decoding happens in collate_fn.
    """

    def __init__(self, hf_split, label2id: dict[int, int], num_labels: int):
        self.ds = hf_split
        self.label2id = label2id
        self.num_labels = num_labels

    def __len__(self):
        return len(self.ds)

    def _extract_video_path(self, ex: dict[str, Any]) -> str:
        # Charades 480p returns a string path in ex["video"]
        if "video" in ex:
            v = ex["video"]
            if isinstance(v, str) and v:
                return v

            # Other HF video datasets might store dict/object
            if isinstance(v, dict) and "path" in v and v["path"]:
                return v["path"]
            if hasattr(v, "path") and getattr(v, "path"):
                return v.path

        # fallback keys
        for k in ("video_path", "path", "filepath", "file"):
            if k in ex and ex[k]:
                return ex[k]

        raise KeyError(f"Could not find video path keys in example: {list(ex.keys())}")

    def _extract_label_ids(self, ex: dict[str, Any]) -> list[int]:
        # Charades stores labels as list[int]
        if "labels" in ex and isinstance(ex["labels"], list):
            return [int(x) for x in ex["labels"]]
        return []

    def __getitem__(self, idx: int):
        ex = self.ds[idx]
        video_path = self._extract_video_path(ex)
        label_ids = self._extract_label_ids(ex)

        # Build multi-hot vector in subset label space
        y = torch.zeros(self.num_labels, dtype=torch.float32)
        for gid in label_ids:
            pos = self.label2id.get(int(gid))
            if pos is not None:
                y[pos] = 1.0

        return {"video_path": video_path, "labels": y}