import os
import cv2
import torch
from transformers import AutoVideoProcessor

from src.train.device import pick_device
from src.models.vjepa2_multilabel import VJEPA2MultiLabel
from src.infer.load import load_diary_actions


# --------- EDIT THESE ----------
VIDEO_PATH = "/Users/ray/clip-video-logger/src/data/raw/video_2.mov" 
SUBSET_JSON = "configs/diary_actions.json"

MODEL_NAME = "facebook/vjepa2-vitl-fpc16-256-ssv2"
FT_CKPT = "outputs/checkpoints/vjepa2_diary_lastblock.pt"  # finetuned checkpoint

OUT_RAW = "outputs/raw_overlay_video_1.mp4"
OUT_FT  = "outputs/ft_overlay_video_1.mp4"

THR = 0.10
TOP_K = 5
STRIDE_SECONDS = 1.0

FRAMES_PER_CLIP = 16
FRAME_STRIDE = 4

DEVICE = "auto"
# ------------------------------


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def overlay_lines(frame, lines, x=20, y=40, line_h=28):
    for i, text in enumerate(lines):
        yy = y + i * line_h
        cv2.putText(
            frame,
            text,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return frame


def sample_clip(cap, start_frame, frames_per_clip, frame_stride):
    frames = []
    for i in range(frames_per_clip):
        fidx = start_frame + i * frame_stride
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(frame)
    return frames


@torch.inference_mode()
def clip_probs(model, processor, device, clip_bgr):
    clip_rgb = [cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) for fr in clip_bgr]
    inputs = processor(clip_rgb, return_tensors="pt")

    if "pixel_values_videos" in inputs:
        pv = inputs["pixel_values_videos"]
    elif "pixel_values" in inputs:
        pv = inputs["pixel_values"]
    else:
        raise KeyError(f"Processor outputs keys: {list(inputs.keys())}")

    pv = pv.to(device)
    logits = model.model(pixel_values_videos=pv).logits  # (1,C)
    probs = torch.sigmoid(logits)[0].detach().cpu()
    return probs


def top_actions(probs, action_names, thr, top_k):
    above = torch.where(probs >= thr)[0].tolist()
    if len(above) == 0:
        topk = torch.topk(probs, k=min(top_k, probs.numel()))
        idxs = topk.indices.tolist()
    else:
        idxs = sorted(above, key=lambda i: float(probs[i]), reverse=True)[:top_k]
    return [f"{action_names[i]} ({float(probs[i]):.2f})" for i in idxs]


def load_model(num_labels, ckpt_path=None):
    device = pick_device(DEVICE)
    processor = AutoVideoProcessor.from_pretrained(MODEL_NAME)

    model = VJEPA2MultiLabel(model_name=MODEL_NAME, num_labels=num_labels, freeze_backbone=False)

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # supports multiple checkpoint formats
        if isinstance(ckpt, dict) and "classifier_state_dict" in ckpt:
            model.model.classifier.load_state_dict(ckpt["classifier_state_dict"], strict=True)
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt, strict=False)
        else:
            raise ValueError(f"Unrecognized checkpoint format: {type(ckpt)}")

    model.to(device).eval()
    return model, processor, device


def render_overlay_video(model, processor, device, out_path, tag, action_names):
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stride_frames = max(1, int(round(STRIDE_SECONDS * fps)))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {out_path}")

    print(f"\n▶️ Rendering {tag} overlay -> {out_path}")
    print(f"device={device} fps={fps:.2f} stride_frames={stride_frames} thr={THR}")

    next_infer = 0
    last_lines = [f"{tag}: ..."]
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if frame_idx >= next_infer:
            clip = sample_clip(cap, frame_idx, FRAMES_PER_CLIP, FRAME_STRIDE)
            if len(clip) >= 2:
                probs = clip_probs(model, processor, device, clip)
                lines = top_actions(probs, action_names, THR, TOP_K)
                last_lines = [f"{tag}  thr={THR:.2f}"] + lines
            else:
                last_lines = [f"{tag}: (clip decode failed)"]
            next_infer = frame_idx + stride_frames

        overlay_lines(frame, last_lines, x=20, y=40, line_h=28)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"✅ Done: {out_path}")


def main():
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(VIDEO_PATH)
    if not os.path.exists(SUBSET_JSON):
        raise FileNotFoundError(SUBSET_JSON)
    if not os.path.exists(FT_CKPT):
        raise FileNotFoundError(FT_CKPT)

    ensure_dir(OUT_RAW)
    ensure_dir(OUT_FT)

    action_names = load_diary_actions(SUBSET_JSON)
    num_labels = len(action_names)

    # RAW: backbone + random head (no ckpt)
    raw_model, processor, device = load_model(num_labels, ckpt_path=None)
    render_overlay_video(raw_model, processor, device, OUT_RAW, tag="RAW", action_names=action_names)

    # FT: load your ckpt
    ft_model, processor2, device2 = load_model(num_labels, ckpt_path=FT_CKPT)
    render_overlay_video(ft_model, processor2, device2, OUT_FT, tag="FINETUNED", action_names=action_names)

    print("\n🎬 Compare these two outputs:")
    print("  -", OUT_RAW)
    print("  -", OUT_FT)


if __name__ == "__main__":
    main()