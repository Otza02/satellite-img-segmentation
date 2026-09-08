import torch
from torch import nn
from torch.nn import functional as F
from satelliteSegmentation.config import Config
from satelliteSegmentation.models._components import Encoder, Decoder, Block


class UNet(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.encoder = Encoder(
            config.in_channels,
            list(config.hidden_channels),
            config.kernel_size,
            config.stride,
        )

        self.bottleneck = Block(
            config.hidden_channels[-1],
            config.bottleneck_channels,
            config.kernel_size,
            config.stride,
        )

        self.decoder = Decoder(
            config.bottleneck_channels,
            list(reversed(config.hidden_channels)),
            config.kernel_size,
            config.stride,
        )

        self.classifier = nn.Conv2d(
            config.hidden_channels[0],
            config.num_classes,
            1
            # config.kernel_size,
            # config.stride,
            # (config.kernel_size - 1) // 2
        )

    def forward(self, x: torch.Tensor):
        x, skip_connections = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_connections)
        return self.classifier(x)


def main():
    conf = Config(device="cpu")
    model = UNet(conf)
    print("Prueba para 120x120")
    x = torch.randn([4, 3, 120, 120])
    with torch.no_grad():
        result = model(x)
    print(f"Imagen entrada: {x.shape}")
    print(f"Imagen salida: {result.shape}")
    assert x.shape[2:] == result.shape[2:]
    
    print("\nPrueba para 128x128")
    x = torch.randn([4, 3, 128, 128])
    with torch.no_grad():
        result = model(x)
    print(f"Imagen entrada: {x.shape}")
    print(f"Imagen salida: {result.shape}")
    assert x.shape[2:] == result.shape[2:]


if __name__ == "__main__":
    main()
