from dataclasses import dataclass, asdict
from torch import Tensor


@dataclass
class Config:
    # General
    device: str

    # Model
    kernel_size: int = 3
    in_channels: int = 3
    hidden_channels: tuple[int, ...] = (64, 128, 256, 512)
    bottleneck_channels: int = 1024
    num_classes: int = 7

    # Train
    epochs: int = 100
    lr: float = 1e-4
    patience: int = 10
    min_delta: float = 1e-3
    weights: Tensor | None = None

    def to_json(self):
        data = asdict(self)
        if self.weights is not None:
            data["weights"] = self.weights.detach().cpu().tolist()
        return data
