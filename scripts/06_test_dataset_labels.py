from datasets import load_dataset
from src.data.diary_subset import load_diary_subset
from src.data.charades import CharadesVideoDataset

def main():
    subset = load_diary_subset("configs/diary_actions.json")

    hf_ds = load_dataset(
        "HuggingFaceM4/charades",
        "480p",
        split="train[:10]",
        trust_remote_code=True,
    )

    ds = CharadesVideoDataset(
        hf_split=hf_ds,
        label2id=subset.id2pos,
        num_labels=len(subset.action_ids),
    )

    ex = ds[0]
    print("video_path:", ex["video_path"])
    print("labels shape:", ex["labels"].shape)
    print("num positive labels:", int(ex["labels"].sum()))
    print("positive label indices:", ex["labels"].nonzero().squeeze(-1).tolist())

if __name__ == "__main__":
    main()
