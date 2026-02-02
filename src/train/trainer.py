from __future__ import annotations
import os
import torch
import torch.nn as nn

from src.train.config import TrainConfig


def build_optimizer(model: nn.Module, cfg: TrainConfig):
    head_params = []
    backbone_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # classifier params
        if ".classifier." in name or name.startswith("model.classifier.") or name.startswith("classifier."):
            head_params.append(p)
        else:
            backbone_params.append(p)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": cfg.head_lr})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cfg.backbone_lr})

    if not param_groups:
        raise RuntimeError("No trainable parameters found. Did you accidentally freeze everything?")

    return torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)


def save_head_only(model: nn.Module, path: str):
    # Save only classifier weights to keep file small-ish and avoid backbone mismatch issues
    state = {"classifier_state_dict": model.model.classifier.state_dict()}
    torch.save(state, path)


def save_full_trainable(model: nn.Module, path: str):
    # Save full model state (big). Useful if you also unfroze blocks.
    torch.save({"model_state_dict": model.state_dict()}, path)

