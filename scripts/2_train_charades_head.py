import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import cycle
from datasets import load_dataset
from transformers import AutoVideoProcessor

from src.train.config import TrainConfig
from src.train.device import pick_device
from src.data.diary_subset import load_diary_subset
from src.data.charades import CharadesVideoDataset
from src.data.collate import VideoCollator
from src.models.vjepa2_multilabel import VJEPA2MultiLabel
from src.train.trainer import build_optimizer, save_head_only, save_full_trainable


def main():
    cfg = TrainConfig()
    device = pick_device(cfg.device)

    print("✅ train script started")
    print("device:", device)
    print("cfg:", cfg)

    subset = load_diary_subset(cfg.subset_path)
    num_labels = len(subset.action_ids)
    cfg.num_labels = num_labels
    print("subset labels:", num_labels)

    print("Loading dataset...")
    train_hf = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.split_train, trust_remote_code=True)
    val_hf = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.split_val, trust_remote_code=True)

    if cfg.max_train_samples is not None:
        train_hf = train_hf.select(range(min(cfg.max_train_samples, len(train_hf))))
    if cfg.max_val_samples is not None:
        val_hf = val_hf.select(range(min(cfg.max_val_samples, len(val_hf))))

    train_ds = CharadesVideoDataset(train_hf, label2id=subset.id2pos, num_labels=num_labels)
    val_ds = CharadesVideoDataset(val_hf, label2id=subset.id2pos, num_labels=num_labels)

    print("train size:", len(train_ds), "val size:", len(val_ds))

    processor = AutoVideoProcessor.from_pretrained(cfg.model_name)
    collate = VideoCollator(
        processor,
        frames_per_clip=cfg.frames_per_clip,
        frame_stride=cfg.frame_stride,
        max_retries=5,
        warn_every=25,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate,
    )

    model = VJEPA2MultiLabel(cfg.model_name, num_labels=num_labels, unfreeze_last_n=cfg.unfreeze_last_n_blocks)
    model.to(device)

    # report trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("model ready. trainable params:", trainable)
    if getattr(model, "unfrozen_blocks", None):
        print("unfrozen blocks:", model.unfrozen_blocks)

    optimizer = build_optimizer(model, cfg)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    step = 0
    optimizer.zero_grad(set_to_none=True)

    print("starting training loop...")
    for batch in cycle(train_loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        out = model.model(pixel_values_videos=batch["pixel_values_videos"])
        logits = out.logits
        loss = loss_fn(logits, batch["labels"])
        loss = loss / cfg.grad_accum_steps
        loss.backward()

        if (step + 1) % cfg.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        step += 1
        if step % cfg.log_every == 0:
            print(f"step {step:04d} | loss {loss.item() * cfg.grad_accum_steps:.4f}")

        if step >= cfg.max_steps:
            break

        if cfg.save_every and step % cfg.save_every == 0:
            os.makedirs(cfg.out_dir, exist_ok=True)
            ckpt_path = os.path.join(cfg.out_dir, "vjepa2_diary_lastblock.pt" if cfg.unfreeze_last_n_blocks > 0 else "vjepa2_diary_head_only.pt")
            if cfg.unfreeze_last_n_blocks > 0:
                save_full_trainable(model, ckpt_path)
                print("✅ saved full-trainable checkpoint:", ckpt_path)
            else:
                save_head_only(model, ckpt_path)
                print("✅ saved head-only checkpoint:", ckpt_path)

    os.makedirs(cfg.out_dir, exist_ok=True)
    final_path = os.path.join(cfg.out_dir, "vjepa2_diary_lastblock.pt" if cfg.unfreeze_last_n_blocks > 0 else "vjepa2_diary_head_only.pt")
    if cfg.unfreeze_last_n_blocks > 0:
        save_full_trainable(model, final_path)
        print("✅ saved full-trainable checkpoint:", final_path)
    else:
        save_head_only(model, final_path)
        print("✅ saved head-only checkpoint:", final_path)

    print("✅ training finished")


if __name__ == "__main__":
    main()

# import os
# import torch
# from torch.utils.data import DataLoader
# from transformers import AutoVideoProcessor

# from datasets import load_dataset

# from src.train.config import TrainConfig
# from src.train.device import pick_device
# from src.data.diary_subset import load_diary_subset
# from src.data.charades import CharadesVideoDataset
# from src.data.collate import VideoCollator
# from src.models.vjepa2_multilabel import VJEPA2MultiLabel


# def main():
#     print("✅ train script started")

#     cfg = TrainConfig()
#     device = pick_device(cfg.device)
#     print("device:", device)
#     print("cfg:", cfg)

#     # ---- load diary subset ----
#     subset = load_diary_subset(cfg.subset_path)
#     num_labels = len(subset.action_ids)
#     print("subset labels:", num_labels)

#     # ---- load HF dataset ----
#     train_hf = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.split_train, trust_remote_code=True)
#     val_hf   = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.split_val, trust_remote_code=True)

#     if cfg.max_train_samples:
#         train_hf = train_hf.select(range(min(cfg.max_train_samples, len(train_hf))))
#     if cfg.max_val_samples:
#         val_hf = val_hf.select(range(min(cfg.max_val_samples, len(val_hf))))

#     print("train size:", len(train_hf), "val size:", len(val_hf))

#     train_ds = CharadesVideoDataset(train_hf, label2id=subset.id2pos, num_labels=num_labels)
#     val_ds   = CharadesVideoDataset(val_hf,   label2id=subset.id2pos, num_labels=num_labels)

#     # ---- processor + collator ----
#     processor = AutoVideoProcessor.from_pretrained(cfg.model_name)
#     collate = VideoCollator(processor, frames_per_clip=cfg.frames_per_clip, frame_stride=cfg.frame_stride)

#     train_loader = DataLoader(
#         train_ds,
#         batch_size=cfg.batch_size,
#         shuffle=True,
#         num_workers=getattr(cfg, "num_workers", 0),
#         collate_fn=collate,
#     )

#     # ---- model ----
#     model = VJEPA2MultiLabel(cfg.model_name, num_labels=num_labels, freeze_backbone=cfg.freeze_backbone)
#     model.to(device)

#     # ---- optimizer (head-only if frozen) ----
#     params = [p for p in model.parameters() if p.requires_grad]
#     optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

#     print("model ready. trainable params:", sum(p.numel() for p in params))
#     print("starting training loop...")

#     model.train()
#     step = 0
#     running = 0.0
#     optim.zero_grad(set_to_none=True)

#     while step < cfg.max_steps:
#         for batch in train_loader:
#             batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
#             loss, logits = model(**batch)
#             (loss / cfg.grad_accum_steps).backward()

#             running += loss.item()
#             if (step + 1) % cfg.grad_accum_steps == 0:
#                 optim.step()
#                 optim.zero_grad(set_to_none=True)

#             if (step + 1) % cfg.log_every == 0:
#                 avg = running / cfg.log_every
#                 print(f"step {step+1:04d} | loss {avg:.4f}")
#                 running = 0.0

#             step += 1
#             if step >= cfg.max_steps:
#                 break
    
#     os.makedirs(cfg.out_dir, exist_ok=True)
#     ckpt_path = os.path.join(cfg.out_dir, "vjepa2_diary_head_only.pt")

#     # Save ONLY the classifier head weights (tiny)
#     head_state = model.model.classifier.state_dict()

#     torch.save(
#         {
#             "model_name": cfg.model_name,
#             "num_labels": num_labels,
#             "subset_path": cfg.subset_path,
#             "classifier_state_dict": head_state,
#         },
#         ckpt_path,
#     )

#     print("✅ saved head-only checkpoint:", ckpt_path)

#     print("✅ training finished")


# if __name__ == "__main__":
#     main()
