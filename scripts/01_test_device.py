from src.train.config import TrainConfig
from src.train.device import pick_device

cfg = TrainConfig()
print("config device:", cfg.device)
print("picked device:", pick_device(cfg.device))
