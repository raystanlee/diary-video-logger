from __future__ import annotations
import torch.nn as nn
from transformers import AutoModelForVideoClassification

from src.models.finetune import freeze_all, unfreeze_module, unfreeze_last_n_blocks


class VJEPA2MultiLabel(nn.Module):
    """
    Wrapper for V-JEPA2 video classification model (multi-label via sigmoid in training/eval scripts).

    We support:
      - head-only tuning: train classifier only
      - last-block tuning: train classifier + last N encoder blocks
    """
    def __init__(self, model_name: str, num_labels: int, unfreeze_last_n: int = 0):
        super().__init__()

        self.model = AutoModelForVideoClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

        # Freeze everything first
        freeze_all(self.model)

        # Always train classifier head
        if hasattr(self.model, "classifier"):
            unfreeze_module(self.model.classifier)
        else:
            raise RuntimeError("Expected model.classifier to exist.")

        # Unfreeze last N blocks of the encoder backbone
        self.unfrozen_blocks = []
        if unfreeze_last_n > 0:
            if not hasattr(self.model, "vjepa2") or not hasattr(self.model.vjepa2, "encoder"):
                raise RuntimeError("Expected model.vjepa2.encoder to exist.")
            self.unfrozen_blocks = unfreeze_last_n_blocks(self.model.vjepa2.encoder, unfreeze_last_n)