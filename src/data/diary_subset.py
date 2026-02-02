import json
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class DiarySubset:
    action_ids: List[int]          # global IDs in Charades (0..156)
    action_names: List[str]        # readable names
    id2pos: Dict[int, int]         # global_id -> position in subset vector

def load_diary_subset(path: str = "configs/diary_actions.json") -> DiarySubset:
    with open(path, "r") as f:
        obj = json.load(f)
    action_ids = obj["action_ids"]
    action_names = obj["action_names"]
    id2pos = {gid: i for i, gid in enumerate(action_ids)}
    return DiarySubset(action_ids=action_ids, action_names=action_names, id2pos=id2pos)
