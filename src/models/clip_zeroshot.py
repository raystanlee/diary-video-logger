from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


@dataclass
class ClipZeroShotResult:
    best_label: str
    best_score: float
    scores: Dict[str, float]


class ClipZeroShot:
    """
    CLIP zero-shot image classifier.
    - Input: PIL Image
    - Output: best label + per-label scores
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        dtype: str = "float16",
    ):
        if device == "auto":
            # mac: mps, else cuda if available, else cpu
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # dtype handling (MPS is happiest with float16/float32 depending on ops)
        if dtype == "float16":
            self.dtype = torch.float16
        elif dtype == "float32":
            self.dtype = torch.float32
        else:
            raise ValueError("dtype must be 'float16' or 'float32'")

        self.model_name = model_name
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)

        # Move model to device
        self.model.to(self.device)
        self.model.eval()

        # Optional: cast model weights (cuda likes fp16; mps can be mixed)
        try:
            self.model.to(dtype=self.dtype)
        except Exception:
            # Safe fallback if dtype casting isn't supported on this backend
            pass

    @torch.no_grad()
    def classify(
        self,
        image: Image.Image,
        labels: List[str],
        template: Optional[str] = None,
    ) -> ClipZeroShotResult:
        """
        labels: list of class names or phrases.
        template: optional prompt template like "a photo of {}"
        """
        if not labels:
            raise ValueError("labels must be a non-empty list")

        if template:
            texts = [template.format(l) for l in labels]
        else:
            texts = labels

        inputs = self.processor(text=texts, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        # logits_per_image shape: [1, num_texts]
        logits = outputs.logits_per_image.squeeze(0)

        # Convert to probabilities (softmax over provided labels)
        probs = torch.softmax(logits, dim=-1)

        scores = {labels[i]: float(probs[i].item()) for i in range(len(labels))}
        best_idx = int(torch.argmax(probs).item())
        best_label = labels[best_idx]
        best_score = scores[best_label]

        return ClipZeroShotResult(best_label=best_label, best_score=best_score, scores=scores)
