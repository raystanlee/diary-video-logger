from __future__ import annotations

from datasets import Dataset


def filter_charades_to_diary(hf_split: Dataset, diary_action_ids: list[int], diary_action_names: list[str] | None = None) -> Dataset:
    """
    Keeps only examples where Charades example.labels (list[int]) intersects diary_action_ids.

    We only need ids. diary_action_names is unused but kept so your call signature doesn't break.
    """
    diary_set = set(int(x) for x in diary_action_ids)

    def keep(ex):
        lbls = ex.get("labels", [])
        if not isinstance(lbls, list):
            return False
        return any(int(x) in diary_set for x in lbls)

    return hf_split.filter(keep)