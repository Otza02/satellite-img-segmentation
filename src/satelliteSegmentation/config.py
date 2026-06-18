from dataclasses import dataclass


@dataclass
class Config:
    # General
    device: str

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
