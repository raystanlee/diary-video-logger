import os
import torch
from transformers import AutoVideoProcessor

from src.data.collate import VideoCollator

VIDEO = "src/data/raw/test_video_1.mov"  # update if needed

def main():
    if not os.path.exists(VIDEO):
        raise FileNotFoundError(f"Video not found: {VIDEO}")

    processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2")
    collate = VideoCollator(processor, frames_per_clip=16, frame_stride=4)

    fake_batch = [{"video_path": VIDEO, "labels": torch.zeros(157)}]
    out = collate(fake_batch)

    print("=== Collate Output ===")
    print("keys:", list(out.keys()))
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: shape={tuple(v.shape)} dtype={v.dtype}")

    # Expected: pixel_values exists (name can vary by model)
    assert any("pixel" in k for k in out.keys()), "Expected a pixel_values-like key in output"
    assert "labels" in out, "Expected labels in output"

    print("✅ Collator + processor looks good.")

if __name__ == "__main__":
    main()
