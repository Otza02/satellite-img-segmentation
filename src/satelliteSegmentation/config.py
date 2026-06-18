from dataclasses import dataclass


@dataclass
class Config:
    # Model
    kernel_size: int = 3
    stride: int = 1
    in_channels: int = 3
    hidden_channels: tuple[int, ...] = (64, 128, 256, 512)
    bottleneck_channels: int = 1024
    num_classes: int = 7
