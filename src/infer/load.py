import json
import os
import torch
from transformers import AutoVideoProcessor

from src.models.vjepa2_multilabel import VJEPA2MultiLabel
from src.train.device import pick_device


def load_diary_actions(path: str) -> list[str]:
    with open(path, "r") as f:
        data = json.load(f)

    if "action_names" not in data:
        raise ValueError("Expected `action_names` in diary_actions.json")

    return data["action_names"]


def load_model_for_infer(
    model_name: str,
    ckpt_path: str,
    num_labels: int,
    device_str: str = "auto",
    freeze_backbone: bool = False,
):
    device = pick_device(device_str)
    processor = AutoVideoProcessor.from_pretrained(model_name)

    model = VJEPA2MultiLabel(model_name=model_name, num_labels=num_labels, freeze_backbone=freeze_backbone)

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Support both checkpoint formats:
    # 1) head-only: {"classifier_state_dict": ...}
    # 2) full: {"model_state_dict": ...} or raw full model state dict
    if isinstance(ckpt, dict) and "classifier_state_dict" in ckpt:
        model.model.classifier.load_state_dict(ckpt["classifier_state_dict"], strict=True)
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif isinstance(ckpt, dict):
        # maybe it IS the state dict
        model.load_state_dict(ckpt, strict=False)
    else:
        raise ValueError(f"Unrecognized checkpoint format at: {ckpt_path}")

    model.to(device).eval()
    return model, processor, device