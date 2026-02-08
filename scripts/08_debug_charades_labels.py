# scripts/08_debug_charades_labels.py
from datasets import load_dataset
from src.train.config import TrainConfig

def main():
    cfg = TrainConfig()
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split="train[:5]", trust_remote_code=True)

    for i, ex in enumerate(ds):
        labels = ex.get("labels", None)
        print(f"\n--- ex[{i}] keys:", list(ex.keys()))
        print("labels type:", type(labels))
        print("labels sample:", labels)
        if isinstance(labels, list) and labels:
            print("labels[0] type:", type(labels[0]))

if __name__ == "__main__":
    main()