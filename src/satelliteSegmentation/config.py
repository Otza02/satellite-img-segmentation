from dataclasses import dataclass
from torch import Tensor


@dataclass
class Config:
    # General
    device: str
    batch_size: int = 128

    # Model
    kernel_size: int = 3
    stride: int = 1
    in_channels: int = 3
    hidden_channels: tuple[int, ...] = (64, 128, 256, 512)
    bottleneck_channels: int = 1024
    num_classes: int = 7

    # Train
    epochs: int = 20
    lr: float = 1e-4
    patience: int = 5
    min_delta: float = 1e-3
    weights: Tensor | None = None
