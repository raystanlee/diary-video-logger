import numpy as np
import torch
from torchcodec.decoders import VideoDecoder

def _framebatch_to_tensor(frames):
    """
    TorchCodec may return a FrameBatch object. This helper extracts
    the underlying tensor across versions.
    """
    # Common attributes in various versions
    for attr in ("data", "frames", "tensor"):
        if hasattr(frames, attr):
            t = getattr(frames, attr)
            return torch.as_tensor(t)

    # Some FrameBatch objects may support .to_list()
    if hasattr(frames, "to_list"):
        lst = frames.to_list()
        return torch.stack([torch.as_tensor(f) for f in lst], dim=0)

    # Fallback: try iterating
    try:
        return torch.stack([torch.as_tensor(f) for f in frames], dim=0)
    except TypeError as e:
        raise TypeError(
            f"Don't know how to convert frames of type {type(frames)}. "
            f"Available attrs: {dir(frames)[:50]}"
        ) from e

def _fetch_frames(vd: VideoDecoder, indices: list[int]) -> torch.Tensor:
    """
    Robust frame fetch across torchcodec versions.
    Returns a tensor [T,H,W,C].
    """
    if hasattr(vd, "get_frames_at"):
        fb = vd.get_frames_at(indices)
        return _framebatch_to_tensor(fb)

    # Fallback: index frames one by one
    frames_list = []
    for i in indices:
        if hasattr(vd, "get_frame"):
            f = vd.get_frame(int(i))
        else:
            f = vd[int(i)]
        frames_list.append(torch.as_tensor(f))
    return torch.stack(frames_list, dim=0)

def sample_frames_torchcodec(
    video_path: str,
    frames_per_clip: int,
    frame_stride: int,
) -> torch.Tensor:
    vd = VideoDecoder(video_path)
    n = len(vd)
    if n <= 0:
        raise ValueError(f"VideoDecoder found 0 frames: {video_path}")

    span = (frames_per_clip - 1) * frame_stride + 1
    start = 0 if n <= span else int(np.random.randint(0, n - span))

    idx = (start + np.arange(frames_per_clip) * frame_stride).clip(0, n - 1).astype(int)
    frames = _fetch_frames(vd, idx.tolist())

    return frames

