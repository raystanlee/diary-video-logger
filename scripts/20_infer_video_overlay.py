import os
import cv2
import torch
from transformers import AutoVideoProcessor

from src.train.device import pick_device
from src.models.vjepa2_multilabel import VJEPA2MultiLabel
from src.infer.load import load_diary_actions


# ----------------------------
# Config (edit these)
# ----------------------------
VIDEO_PATH = "/Users/ray/clip-video-logger/src/data/raw/video_2.mov"
OUT_PATH = "outputs/infer_overlay_video_3.mp4"

MODEL_NAME = "facebook/vjepa2-vitl-fpc16-256-ssv2"
CKPT_PATH = "outputs/checkpoints/handpicked_labelset/vjepa2_diary_lastblock.pt"
SUBSET_JSON = "configs/diary_actions.json"

THRESH = 0.10
TOP_K = 5
STRIDE_SECONDS = 1.0
FRAMES_PER_CLIP = 16
FRAME_STRIDE = 4
DEVICE = "auto"


# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def sample_clip(cap_sample: cv2.VideoCapture, start_frame: int, frames_per_clip: int, frame_stride: int):
    """
    Sample frames using a *separate* VideoCapture so we don't disturb the main streaming reader.
    Returns list of BGR uint8 frames.
    """
    frames = []
    for i in range(frames_per_clip):
        fidx = start_frame + i * frame_stride
        cap_sample.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap_sample.read()
        if not ok or frame is None:
            break
        frames.append(frame)
    return frames


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


@torch.inference_mode()
def main():
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(VIDEO_PATH)
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(CKPT_PATH)
    if not os.path.exists(SUBSET_JSON):
        raise FileNotFoundError(SUBSET_JSON)

    ensure_dir(OUT_PATH)

    action_names = load_diary_actions(SUBSET_JSON)  # expects action_names list in index order
    num_labels = len(action_names)

    device = pick_device(DEVICE)
    processor = AutoVideoProcessor.from_pretrained(MODEL_NAME)

    # Build model (must match your class signature)
    # If you trained last-block + head, we don't need to "unfreeze" at inference
    model = VJEPA2MultiLabel(model_name=MODEL_NAME, num_labels=num_labels, unfreeze_last_n_blocks=0)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")

    # Load checkpoint (supports multiple formats)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    elif isinstance(ckpt, dict) and "classifier_state_dict" in ckpt:
        # head-only checkpoint
        model.model.classifier.load_state_dict(ckpt["classifier_state_dict"], strict=True)
    elif isinstance(ckpt, dict):
        # raw state_dict
        model.load_state_dict(ckpt, strict=False)
    else:
        raise ValueError(f"Unrecognized checkpoint format: {type(ckpt)}")

    model.to(device).eval()

    # Main sequential reader
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    # Separate sampler reader
    cap_sample = cv2.VideoCapture(VIDEO_PATH)
    if not cap_sample.isOpened():
        raise RuntimeError(f"Could not open sampler capture: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_PATH, fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {OUT_PATH}")

    print(f"✅ Running inference on: {VIDEO_PATH}")
    print(f"✅ Saving overlay video to: {OUT_PATH}")
    print(f"device: {device} | fps: {fps:.2f} | stride_seconds: {STRIDE_SECONDS} | thr: {THRESH}")
    print(f"total_frames: {total_frames} | frames_per_clip: {FRAMES_PER_CLIP} stride: {FRAME_STRIDE}")

    stride_frames = max(1, int(round(STRIDE_SECONDS * fps)))
    next_infer_frame = 0
    last_lines = ["..."]

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if frame_idx >= next_infer_frame:
            clip = sample_clip(
                cap_sample,
                start_frame=frame_idx,
                frames_per_clip=FRAMES_PER_CLIP,
                frame_stride=FRAME_STRIDE,
            )

            if len(clip) == FRAMES_PER_CLIP:
                clip_rgb = [cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) for fr in clip]
                inputs = processor(clip_rgb, return_tensors="pt")

                if "pixel_values_videos" in inputs:
                    pv = inputs["pixel_values_videos"]
                elif "pixel_values" in inputs:
                    pv = inputs["pixel_values"]
                else:
                    raise KeyError(f"Processor outputs keys: {list(inputs.keys())}")

                pv = pv.to(device)

                logits = model.model(pixel_values_videos=pv).logits  # (1, C)
                probs = torch.sigmoid(logits)[0].detach().cpu()      # (C,)

                above = torch.where(probs >= THRESH)[0].tolist()
                if len(above) == 0:
                    topk = torch.topk(probs, k=min(TOP_K, probs.numel()))
                    idxs = topk.indices.tolist()
                else:
                    idxs = sorted(above, key=lambda i: float(probs[i]), reverse=True)[:TOP_K]

                last_lines = [f"{action_names[i]} ({float(probs[i]):.2f})" for i in idxs]
                if len(last_lines) == 0:
                    last_lines = ["(no confident action)"]
            else:
                # Near end of video; keep last prediction
                pass

            next_infer_frame = frame_idx + stride_frames

        overlay_lines(frame, last_lines, x=20, y=40, line_h=28)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    cap_sample.release()
    writer.release()
    print("✅ Done. Output:", OUT_PATH)


if __name__ == "__main__":
    main()

# import os
# import cv2
# import json
# import math
# import torch
# from transformers import AutoVideoProcessor

# from src.train.device import pick_device
# from src.models.vjepa2_multilabel import VJEPA2MultiLabel
# from src.infer.load import load_diary_actions


# # ----------------------------
# # Config (edit these)
# # ----------------------------
# VIDEO_PATH = "/Users/ray/clip-video-logger/src/data/raw/video_2.MOV"
# OUT_PATH = "outputs/infer_overlay_video_3.mp4"

# MODEL_NAME = "facebook/vjepa2-vitl-fpc16-256-ssv2"
# CKPT_PATH = "outputs/checkpoints/handpicked_labelset/vjepa2_diary_lastblock.pt"
# # "outputs/checkpoints/vjepa2_diary_lastblock.pt"   # <- your best checkpoint
# SUBSET_JSON = "configs/diary_actions.json"

# THRESH = 0.10              # <- from your tuned threshold
# TOP_K = 5                  # show up to 5 actions
# STRIDE_SECONDS = 1.0       # run inference every 1s
# FRAMES_PER_CLIP = 16
# FRAME_STRIDE = 4           # same as training
# RESIZE_SHORT = 256
# DEVICE = "auto"


# # ----------------------------
# # Helpers
# # ----------------------------
# def ensure_dir(path: str):
#     d = SDK_dir = os.path.dirname(path)
#     if d:
#         os.makedirs(d, exist_ok=True)


# def sample_clip_from_capture(
#     cap: cv2.VideoCapture,
#     start_frame: int,
#     frames_per_clip: int,
#     frame_stride: int,
# ) -> list:
#     """
#     Returns list of frames (BGR uint8) sampled at indices:
#       start_frame + i*frame_stride
#     """
#     frames = []
#     for i in range(frames_per_clip):
#         fidx = start_frame + i * frame_stride
#         cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
#         ok, frame = cap.read()
#         if not ok or frame is None:
#             break
#         frames.append(frame)
#     return frames


# def overlay_lines(frame, lines, x=20, y=40, line_h=28):
#     """
#     Draws semi-readable overlay text.
#     """
#     for i, text in enumerate(lines):
#         yy = y + i * line_h
#         cv2.putText(
#             frame,
#             text,
#             (x, yy),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (255, 255, 255),
#             2,
#             cv2.LINE_AA,
#         )
#     return frame


# @torch.inference_mode()
# def main():
#     if not os.path.exists(VIDEO_PATH):
#         raise FileNotFoundError(VIDEO_PATH)
#     if not os.path.exists(CKPT_PATH):
#         raise FileNotFoundError(CKPT_PATH)
#     if not os.path.exists(SUBSET_JSON):
#         raise FileNotFoundError(SUBSET_JSON)

#     ensure_dir(OUT_PATH)

#     action_names = load_diary_actions(SUBSET_JSON)
#     num_labels = len(action_names)

#     device = pick_device(DEVICE)
#     processor = AutoVideoProcessor.from_pretrained(MODEL_NAME)

#     # load model
#     model = VJEPA2MultiLabel(model_name=MODEL_NAME, num_labels=num_labels, freeze_backbone=False)
#     ckpt = torch.load(CKPT_PATH, map_location="cpu")

#     # supports full checkpoint OR head-only checkpoint formats
#     if isinstance(ckpt, dict) and "classifier_state_dict" in ckpt:
#         model.model.classifier.load_state_dict(ckpt["classifier_state_dict"], strict=True)
#     elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
#         model.load_state_dict(ckpt["model_state_dict"], strict=False)
#     elif isinstance(ckpt, dict):
#         model.load_state_dict(ckpt, strict=False)
#     else:
#         raise ValueError(f"Unrecognized checkpoint format: {type(ckpt)}")

#     model.to(device).eval()

#     # open video
#     cap = cv2.VideoCapture(VIDEO_PATH)
#     if not cap.isOpened():
#         raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

#     fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     writer = cv2.VideoWriter(OUT_PATH, fourcc, fps, (W, H))
#     if not writer.isOpened():
#         raise RuntimeError(f"Could not open writer: {OUT_PATH}")

#     print(f"✅ Running inference on: {VIDEO_PATH}")
#     print(f"✅ Saving overlay video to: {OUT_PATH}")
#     print(f"device: {device} | fps: {fps:.2f} | stride_seconds: {STRIDE_SECONDS} | thr: {THRESH}")

#     stride_frames = max(1, int(round(STRIDE_SECONDS * fps)))

#     # we will compute predictions at these frames
#     next_infer_frame = 0
#     last_lines = ["..."]

#     frame_idx = 0
#     while True:
#         ok, frame = cap.read()
#         if not ok or frame is None:
#             break

#         # run inference at stride
#         if frame_idx >= next_infer_frame:
#             clip = sample_clip_from_capture(
#                 cap,
#                 start_frame=frame_idx,
#                 frames_per_clip=FRAMES_PER_CLIP,
#                 frame_stride=FRAME_STRIDE,
#             )

#             if len(clip) >= 2:
#                 # processor expects list of frames, usually RGB
#                 clip_rgb = [cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) for fr in clip]

#                 inputs = processor(
#                     clip_rgb,
#                     return_tensors="pt",
#                 )

#                 # FIX: do NOT use `or` on tensors
#                 if "pixel_values_videos" in inputs:
#                     pv = inputs["pixel_values_videos"]
#                 elif "pixel_values" in inputs:
#                     pv = inputs["pixel_values"]
#                 else:
#                     raise KeyError(f"Processor outputs keys: {list(inputs.keys())}")

#                 pv = pv.to(device)

#                 out = model.model(pixel_values_videos=pv)
#                 logits = out.logits  # (1, C)
#                 probs = torch.sigmoid(logits)[0].detach().cpu()  # (C,)

#                 # pick actions above threshold and top-k
#                 above = torch.where(probs >= THRESH)[0].tolist()
#                 if len(above) == 0:
#                     # fallback: top-k anyway
#                     topk = torch.topk(probs, k=min(TOP_K, probs.numel()))
#                     idxs = topk.indices.tolist()
#                 else:
#                     # sort above by probability desc and keep top-k
#                     idxs = sorted(above, key=lambda i: float(probs[i]), reverse=True)[:TOP_K]

#                 last_lines = []
#                 for i in idxs:
#                     name = action_names[i]
#                     p = float(probs[i])
#                     last_lines.append(f"{name} ({p:.2f})")

#                 if len(last_lines) == 0:
#                     last_lines = ["(no confident action)"]

#             # schedule next inference
#             next_infer_frame = frame_idx + stride_frames

#         # overlay latest prediction onto every frame
#         overlay_lines(frame, last_lines, x=20, y=40, line_h=28)
#         writer.write(frame)

#         frame_idx += 1

#     cap.release()
#     writer.release()
#     print("✅ Done. Output:", OUT_PATH)


# if __name__ == "__main__":
#     main()