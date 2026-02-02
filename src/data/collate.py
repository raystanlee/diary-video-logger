from __future__ import annotations

import torch
from src.video.sampling import sample_frames_torchcodec


class VideoCollator:
    """
    Collate function that:
    - decodes frames from video_path
    - applies AutoVideoProcessor
    - returns model-ready tensors

    Robustness:
    - Some dataset entries may point to missing or broken videos.
    - We retry a few times so a single bad sample doesn't kill training.
    """

    def __init__(
        self,
        processor,
        frames_per_clip: int = 16,
        frame_stride: int = 4,
        max_retries: int = 5,
        warn_every: int = 10,
    ):
        self.processor = processor
        self.frames_per_clip = frames_per_clip
        self.frame_stride = frame_stride
        self.max_retries = max_retries
        self.warn_every = warn_every
        self._warn_count = 0

    def __call__(self, batch):
        """
        batch: list of dicts with keys: video_path (str), labels (Tensor[num_labels])
        returns: dict with pixel_values_videos + labels
        """
        last_err = None

        # With batch_size=1, batch has one item.
        # We retry decoding the same item a few times and fail with a clear error.
        for attempt in range(1, self.max_retries + 1):
            item = batch[0]
            video_path = item["video_path"]

            try:
                frames = sample_frames_torchcodec(
                    video_path,
                    frames_per_clip=self.frames_per_clip,
                    frame_stride=self.frame_stride,
                )

                # frames expected: [T,C,H,W] uint8
                # Convert to list of frames in HWC for processor stability
                if frames.ndim == 4 and frames.shape[1] in (3, 4):  # TCHW -> THWC
                    frames_thwc = frames.permute(0, 2, 3, 1).contiguous()
                else:
                    frames_thwc = frames

                # HF processor expects list of frames (each HWC)
                frame_list = [frames_thwc[i] for i in range(frames_thwc.shape[0])]

                inputs = self.processor(frame_list, return_tensors="pt")
                inputs["labels"] = item["labels"].unsqueeze(0)

                return inputs

            except Exception as e:
                last_err = e
                self._warn_count += 1
                if self._warn_count % self.warn_every == 0:
                    print(f"[warn] decode failed ({self._warn_count}): {video_path} | {type(e).__name__}: {e}")
                continue

        raise RuntimeError(
            f"Failed to decode video after {self.max_retries} retries. "
            f"Last error: {type(last_err).__name__}: {last_err}"
        )