# from dataclasses import dataclass
# from typing import Optional

# @dataclass
# class TrainConfig:
#     # Dataset
#     dataset_name: str = "HuggingFaceM4/charades"
#     dataset_config: str = "480p"     # IMPORTANT: use the same config you downloaded
#     split_train: str = "train"
#     split_val: str = "test"          # Charades uses "test" on HF

#     max_train_samples: Optional[int] = 2000
#     max_val_samples: Optional[int] = 500

#     # Diary subset
#     subset_path: str = "configs/diary_actions.json"  # produced by your subset script

#     # Video sampling
#     frames_per_clip: int = 16
#     frame_stride: int = 4
#     resize_short: int = 256

#     # Model
#     model_name: str = "facebook/vjepa2-vitl-fpc16-256-ssv2"
#     freeze_backbone: bool = True
#     num_labels: int = 80             # diary subset size

#     # Training
#     device: str = "auto"
#     batch_size: int = 1
#     grad_accum_steps: int = 16
#     lr: float = 1e-4
#     weight_decay: float = 0.0
#     max_steps: int = 2000             # first run: quick proof it learns
#     log_every: int = 20

#     # Dataloader (Mac-friendly)
#     num_workers: int = 0             # keep 0 first; increase later

#     # Output
#     out_dir: str = "outputs/checkpoints"
#     save_every: int = 200

from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainConfig:
    # Dataset
    dataset_name: str = "HuggingFaceM4/charades"
    dataset_config: str = "480p"
    split_train: str = "train"
    split_val: str = "test"
    # max_train_samples: Optional[int] = 2000
    max_train_samples: Optional[int] = None   # use FULL train split
    max_val_samples: Optional[int] = 1000  
    
    # max_val_samples: Optional[int] = 500
    subset_path: str = "configs/diary_actions.json"

    # Video sampling
    frames_per_clip: int = 16
    frame_stride: int = 4
    resize_short: int = 256

    # Model
    model_name: str = "facebook/vjepa2-vitl-fpc16-256-ssv2"
    num_labels: int = 80  # diary subset
    freeze_backbone: bool = True  # kept for backward compat

    # Fine-tuning depth
    # 0 = head-only (what you did)
    # 1 = head + last transformer block (next)
    # 2 = head + last 2 blocks, ...
    unfreeze_last_n_blocks: int = 1

    # Training
    device: str = "auto"
    batch_size: int = 1
    grad_accum_steps: int = 16
    head_lr: float = 1e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 0.0
    max_steps: int = 4000
    log_every: int = 20
    num_workers: int = 0

    # Output
    out_dir: str = "outputs/checkpoints"
    save_every: int = 500