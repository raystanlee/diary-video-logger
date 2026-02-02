import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoVideoProcessor

from src.train.config import TrainConfig
from src.train.device import pick_device
from src.train.metrics import micro_f1_multilabel
from src.data.diary_subset import load_diary_subset
from src.data.charades import CharadesVideoDataset
from src.data.collate import VideoCollator
from src.models.vjepa2_multilabel import VJEPA2MultiLabel


@torch.inference_mode()
def main():
    cfg = TrainConfig()
    device = pick_device(cfg.device)

    # ✅ tune threshold for your most recent checkpoint
    ckpt_path = os.path.join(cfg.out_dir, "vjepa2_diary_lastblock.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    subset = load_diary_subset(cfg.subset_path)
    num_labels = len(subset.action_ids)

    print("Loading eval split...")
    hf = load_dataset(
        cfg.dataset_name,
        cfg.dataset_config,
        split=cfg.split_val,
        trust_remote_code=True,
    )

    n = min(getattr(cfg, "max_val_samples", 1000), len(hf))
    hf = hf.select(range(n))

    ds = CharadesVideoDataset(hf, label2id=subset.id2pos, num_labels=num_labels)

    processor = AutoVideoProcessor.from_pretrained(cfg.model_name)
    collate = VideoCollator(
        processor,
        frames_per_clip=cfg.frames_per_clip,
        frame_stride=cfg.frame_stride,
        max_retries=5,
        warn_every=25,
    )

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=getattr(cfg, "num_workers", 0),
        collate_fn=collate,
    )

    print("Loading model + checkpoint (full model_state_dict)...")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # IMPORTANT: instantiate wrapper exactly how your class expects
    # (no freeze_backbone kwarg)
    model = VJEPA2MultiLabel(cfg.model_name, num_labels=num_labels)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()

    # collect probs + targets
    all_probs = []
    all_tgts = []

    for i, batch in enumerate(loader, 1):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        logits = model.model(pixel_values_videos=batch["pixel_values_videos"]).logits
        probs = torch.sigmoid(logits).detach().cpu()

        all_probs.append(probs)
        all_tgts.append(batch["labels"].detach().cpu())

        if i % 50 == 0:
            print(f"processed {i}/{len(loader)}")

    P = torch.cat(all_probs, dim=0)  # [N, C]
    T = torch.cat(all_tgts, dim=0)   # [N, C]

    # sweep thresholds
    best_thr = None
    best_score = -1.0

    thresholds = [round(x, 2) for x in torch.linspace(0.05, 0.95, steps=19).tolist()]
    for thr in thresholds:
        score = micro_f1_multilabel(T, P, threshold=thr)
        print(f"thr={thr:.2f} micro-F1={score:.4f}")
        if score > best_score:
            best_score = score
            best_thr = thr

    print(f"\n✅ BEST threshold: {best_thr:.2f}  micro-F1={best_score:.4f}")
    print(f"Evaluated on N={T.shape[0]} videos, C={T.shape[1]} labels")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()

# import os
# import torch
# from torch.utils.data import DataLoader
# from datasets import load_dataset
# from transformers import AutoVideoProcessor

# from src.train.config import TrainConfig
# from src.train.device import pick_device
# from src.data.diary_subset import load_diary_subset
# from src.data.charades import CharadesVideoDataset
# from src.data.collate import VideoCollator
# from src.models.vjepa2_multilabel import VJEPA2MultiLabel


# def f1_micro(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-9) -> float:
#     tp = (preds * targets).sum().item()
#     fp = (preds * (1 - targets)).sum().item()
#     fn = ((1 - preds) * targets).sum().item()
#     return (2 * tp) / (2 * tp + fp + fn + eps)


# @torch.inference_mode()
# def main():
#     cfg = TrainConfig()
#     device = pick_device(cfg.device)

#     ckpt_path = os.path.join(cfg.out_dir, "vjepa2_diary_lastblock.pt")
#     # ckpt_path = os.path.join(cfg.out_dir, "vjepa2_diary_head_only.pt")
#     if not os.path.exists(ckpt_path):
#         raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

#     subset = load_diary_subset(cfg.subset_path)
#     num_labels = len(subset.action_ids)

#     print("Loading eval split...")
#     hf = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.split_val, trust_remote_code=True)
#     hf = hf.select(range(min(200, len(hf))))
#     ds = CharadesVideoDataset(hf, label2id=subset.id2pos, num_labels=num_labels)

#     processor = AutoVideoProcessor.from_pretrained(cfg.model_name)
#     collate = VideoCollator(
#         processor,
#         frames_per_clip=cfg.frames_per_clip,
#         frame_stride=cfg.frame_stride,
#         max_retries=5,
#         warn_every=25,
#     )

#     loader = DataLoader(
#         ds,
#         batch_size=1,
#         shuffle=False,
#         num_workers=getattr(cfg, "num_workers", 0),
#         collate_fn=collate,
#     )

#     print("Loading model + head-only weights...")
#     ckpt = torch.load(ckpt_path, map_location="cpu")
#     # model = VJEPA2MultiLabel(cfg.model_name, num_labels=num_labels, freeze_backbone=True)
#     # model.model.classifier.load_state_dict(ckpt["classifier_state_dict"], strict=True)
#     model = VJEPA2MultiLabel(cfg.model_name, num_labels=num_labels, freeze_backbone=False, unfreeze_last_n=0)
#     model.load_state_dict(ckpt["model_state_dict"], strict=True)
#     model.to(device).eval()

#     all_probs, all_tgts = [], []
#     for i, batch in enumerate(loader, 1):
#         batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
#         logits = model.model(pixel_values_videos=batch["pixel_values_videos"]).logits
#         probs = torch.sigmoid(logits).cpu()
#         all_probs.append(probs)
#         all_tgts.append(batch["labels"].cpu())
#         if i % 50 == 0:
#             print(f"processed {i}/{len(loader)}")

#     P = torch.cat(all_probs, 0)
#     T = torch.cat(all_tgts, 0)

#     best_thr, best = None, -1.0
#     # wider sweep
#     thresholds = [round(x, 2) for x in torch.linspace(0.05, 0.95, steps=19).tolist()]
#     for thr in thresholds:
#         preds = (P >= thr).float()
#         score = f1_micro(preds, T)
#         print(f"thr={thr:.2f} micro-F1={score:.4f}")
#         if score > best:
#             best = score
#             best_thr = thr

#     print(f"\n✅ BEST threshold: {best_thr:.2f}  micro-F1={best:.4f}")


# if __name__ == "__main__":
#     main()