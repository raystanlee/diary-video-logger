import os
import torch
from src.video.sampling import sample_frames_torchcodec

# Change this to any local video path you have
VIDEO = "src/data/raw/test_video_1.mov"  # <-- update if needed

def main():
    if not os.path.exists(VIDEO):
        raise FileNotFoundError(f"Video not found: {VIDEO}. Update VIDEO path in this script.")

    frames = sample_frames_torchcodec(VIDEO, frames_per_clip=8, frame_stride=4)

    print("=== Sampling Test ===")
    print("video:", VIDEO)
    print("frames shape:", tuple(frames.shape))
    print("dtype:", frames.dtype)
    print("min/max:", int(frames.min()), int(frames.max()))
    print("is contiguous:", frames.is_contiguous())

    # basic sanity checks
    assert frames.ndim == 4, "Expected 4D tensor"

    # Accept either [T,H,W,C] or [T,C,H,W]
    if frames.shape[-1] in (3, 4):
        layout = "THWC"
        assert frames.shape[0] == 8, "Expected T=8 frames"
        assert frames.shape[-1] in (3, 4), "Expected last dim to be 3 (RGB) or 4 (RGBA)"
    elif frames.shape[1] in (3, 4):
        layout = "TCHW"
        assert frames.shape[0] == 8, "Expected T=8 frames"
        assert frames.shape[1] in (3, 4), "Expected channel dim to be 3 (RGB) or 4 (RGBA)"
    else:
        raise AssertionError(f"Unknown frame layout: shape={tuple(frames.shape)}")

    print("layout:", layout)

    assert frames.dtype in (torch.uint8, torch.int16, torch.int32, torch.float32), "Unexpected dtype"

    print("✅ Sampling looks good.")

if __name__ == "__main__":
    main()
