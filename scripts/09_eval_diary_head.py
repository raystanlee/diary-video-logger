import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoVideoProcessor

from src.train.config import TrainConfig
from src.train.device import pick_device
from src.data.diary_subset import load_diary_subset
from src.data.charades import CharadesVideoDataset
from src.data.collate import VideoCollator
from src.models.vjepa2_multilabel import VJEPA2MultiLabel
from src.train.metrics import micro_f1_multilabel  # assuming you already have this


def main():
    cfg = TrainConfig()
    device = pick_device(cfg.device)

    subset = load_diary_subset(cfg.subset_path)
    num_labels = len(subset.action_ids)

    # pick which checkpoint to evaluate
    ckpt_path = os.path.join(cfg.out_dir, "vjepa2_diary_lastblock.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print("Loading val split...")
    val_hf = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.split_val, trust_remote_code=True)
    val_hf = val_hf.select(range(min(cfg.max_val_samples or 100, len(val_hf))))
    print("val examples:", len(val_hf))

    val_ds = CharadesVideoDataset(val_hf, label2id=subset.id2pos, num_labels=num_labels)

    processor = AutoVideoProcessor.from_pretrained(cfg.model_name)
    collate = VideoCollator(
        processor,
        frames_per_clip=cfg.frames_per_clip,
        frame_stride=cfg.frame_stride,
        max_retries=5,
        warn_every=25,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    print("Loading model + checkpoint...")
    model = VJEPA2MultiLabel(cfg.model_name, num_labels=num_labels, unfreeze_last_n=0)  # eval: we don't need unfreezing
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    print("Model loaded. device:", device)

    y_true = []
    y_prob = []

    print("Starting eval loop...")
    with torch.no_grad():
        for i, batch in enumerate(val_loader, start=1):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            out = model.model(pixel_values_videos=batch["pixel_values_videos"])
            logits = out.logits
            probs = torch.sigmoid(logits).detach().cpu()

            y_prob.append(probs[0])
            y_true.append(batch["labels"].detach().cpu()[0])

            if i % 10 == 0:
                print(f"  processed {i}/{len(val_loader)}")

    y_true = torch.stack(y_true, dim=0)
    y_prob = torch.stack(y_prob, dim=0)

    thr = 0.10  # use your tuned threshold, or change this
    f1 = micro_f1_multilabel(y_true, y_prob, threshold=thr)

    print()
    print(f"✅ micro-F1 @thr={thr}: {f1:.4f}")
    print("num examples:", y_true.shape[0], "num_labels:", y_true.shape[1])


if __name__ == "__main__":
    main()
