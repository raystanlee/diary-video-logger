import os
import torch
from transformers import AutoVideoProcessor

from src.data.collate import VideoCollator
from src.models.vjepa2_multilabel import VJEPA2MultiLabel
from src.train.device import pick_device

VIDEO = "src/data/raw/test_video_1.mov"  # update if needed
MODEL = "facebook/vjepa2-vitl-fpc16-256-ssv2"

def main():
    if not os.path.exists(VIDEO):
        raise FileNotFoundError(f"Video not found: {VIDEO}")

    device = pick_device("auto")
    print("device:", device)

    processor = AutoVideoProcessor.from_pretrained(MODEL)
    collate = VideoCollator(processor, frames_per_clip=16, frame_stride=4)

    batch = collate([{"video_path": VIDEO, "labels": torch.zeros(157)}])

    # Move tensors to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    model = VJEPA2MultiLabel(MODEL, num_labels=157, freeze_backbone=True)
    model.to(device)

    loss, logits = model(**batch)

    print("loss:", float(loss.item()))
    print("logits shape:", tuple(logits.shape))
    print("logits dtype:", logits.dtype)

    assert logits.shape == (1, 157), "Expected logits shape (B, num_labels)"
    print("✅ Forward pass works.")

if __name__ == "__main__":
    main()
