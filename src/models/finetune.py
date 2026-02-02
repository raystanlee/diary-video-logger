from __future__ import annotations
from typing import Iterable, List, Optional, Tuple
import torch.nn as nn


def freeze_all(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


def _find_block_container(encoder: nn.Module) -> Tuple[str, nn.Module, int]:
    """
    Try to find the transformer block list inside the encoder.

    Returns:
      (attr_name, container_module, n_blocks)

    Supports common patterns:
      encoder.blocks (ModuleList)
      encoder.layers (ModuleList)
      encoder.encoder.layers
      encoder.transformer.layers
    """
    candidates = [
        "blocks",
        "layers",
        "encoder.layers",
        "transformer.layers",
        "backbone.layers",
    ]

    def get_by_path(root: nn.Module, path: str) -> Optional[nn.Module]:
        cur = root
        for part in path.split("."):
            if not hasattr(cur, part):
                return None
            cur = getattr(cur, part)
        return cur

    for path in candidates:
        obj = get_by_path(encoder, path)
        if obj is None:
            continue
        if isinstance(obj, (nn.ModuleList, list, tuple)) and len(obj) > 0:
            return path, obj, len(obj)

    # Fallback: search recursively for a ModuleList that *looks like* blocks
    for name, child in encoder.named_modules():
        if isinstance(child, nn.ModuleList) and len(child) > 0:
            # Heuristic: block modules often have attention/mlp inside
            sample = child[-1]
            sample_names = [n.lower() for n, _ in sample.named_modules()]
            if any("attn" in n or "attention" in n for n in sample_names) and any("mlp" in n or "ff" in n for n in sample_names):
                return name, child, len(child)

    raise RuntimeError(
        "Could not find transformer block container inside encoder. "
        "Please print encoder structure to identify blocks/layers."
    )


def unfreeze_last_n_blocks(encoder: nn.Module, n: int) -> List[str]:
    """
    Unfreeze only the last n blocks of the encoder.
    Returns a list of block identifiers that were unfrozen.
    """
    if n <= 0:
        return []

    path, blocks, total = _find_block_container(encoder)
    n = min(n, total)

    unfrozen = []
    for i in range(total - n, total):
        block = blocks[i]
        unfreeze_module(block)
        unfrozen.append(f"{path}[{i}]")

    return unfrozen