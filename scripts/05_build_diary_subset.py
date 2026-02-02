import json
import os
from datasets import load_dataset

DATASET = "HuggingFaceM4/charades"
CONFIG = "480p"  # matches what you're downloading/using

# Keywords to pick "apartment diary" actions
KEYWORDS = [
    "door", "closet", "cabinet", "refrigerator", "window", "drawer",
    "table", "chair", "sofa", "couch", "bed", "floor",
    "laptop", "computer", "phone", "paper", "notebook",
    "drink", "cup", "bottle", "eat", "sandwich", "cook",
    "wash", "clean", "wipe", "hands", "dish",
    "walk", "run", "standing", "sitting", "lying",
    "light", "television", "vacuum",
]

def main():
    os.makedirs("configs", exist_ok=True)

    # Load a tiny slice just to get features (no need to scan examples)
    ds = load_dataset(DATASET, CONFIG, split="train[:1]", trust_remote_code=True)

    # Human-readable action names live here
    action_names = ds.features["labels"].feature.names  # list[str], length 157
    print(f"Total Charades actions: {len(action_names)}")

    diary_action_ids = []
    diary_action_names = []

    for action_id, name in enumerate(action_names):
        name_l = name.lower()
        if any(k in name_l for k in KEYWORDS):
            diary_action_ids.append(action_id)
            diary_action_names.append(name)

    print(f"Diary subset size: {len(diary_action_ids)}")

    out = {
        "action_ids": diary_action_ids,      # ints you will train on
        "action_names": diary_action_names,  # readable strings for logs
    }

    path = "configs/diary_actions.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print("Saved:", path)
    print("\nSample diary actions:")
    for aid, nm in list(zip(diary_action_ids, diary_action_names))[:25]:
        print(f"{aid:3d}: {nm}")

if __name__ == "__main__":
    main()
