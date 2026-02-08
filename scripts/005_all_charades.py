import os
from datasets import load_dataset

DATASET = "HuggingFaceM4/charades"
CONFIG = "480p"

def main():
    os.makedirs("configs", exist_ok=True)

    ds = load_dataset(DATASET, CONFIG, split="train[:1]", trust_remote_code=True)
    action_names = ds.features["labels"].feature.names  # length 157

    path = "configs/charades_labels_for_editing.txt"
    with open(path, "w") as f:
        for i, name in enumerate(action_names):
            # tab-separated makes it easy to copy into Sheets too
            f.write(f"{i}\t{name}\n")

    print(f"Wrote {len(action_names)} labels to {path}")
    print("Tip: open the file, delete rows you don't want, keep the rest.")

if __name__ == "__main__":
    main()
