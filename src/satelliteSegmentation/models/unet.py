import torch
from torch import nn
from torch.nn import functional as F
from satelliteSegmentation.config import Config


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()

        pad = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, pad)

        self.batch_norm_1 = nn.BatchNorm2d(out_channels)
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.batch_norm_1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batch_norm_2(x)
        return self.relu(x)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channel: int,
        hidden_channels: list[int],
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        last_channel = in_channel
        for channel in hidden_channels:
            self.blocks.append(Block(last_channel, channel, kernel_size, stride))
            last_channel = channel

    def forward(self, x: torch.Tensor):
        skip_connections: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            skip_connections.append(x)
            x = self.pool(x)

        return x, skip_connections


class Decoder(nn.Module):
    def __init__(
        self,
        in_channel: int,
        hidden_channels: list[int],
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        self.up_pools = nn.ModuleList()

        last_channel = in_channel
        for channel in hidden_channels:
            self.up_pools.append(nn.ConvTranspose2d(last_channel, channel, 2, 2))
            self.blocks.append(Block(channel * 2, channel, kernel_size, stride))
            last_channel = channel

    def forward(self, x: torch.Tensor, skip_conn: list[torch.Tensor]):
        for up, block, skip in zip(self.up_pools, self.blocks, reversed(skip_conn)):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:])

            x = torch.cat([x, skip], dim=1)
            x = block(x)

        return x


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
    x = torch.randn([1, 3, 120, 120])
    with torch.no_grad():
        result = model(x)
    print(f"Imagen entrada: {x.shape}")
    print(f"Imagen salida: {result.shape}")
    assert x.shape[2:] == result.shape[2:]
    
    print("\nPrueba para 128x128")
    x = torch.randn([1, 3, 128, 128])
    with torch.no_grad():
        result = model(x)
    print(f"Imagen entrada: {x.shape}")
    print(f"Imagen salida: {result.shape}")
    assert x.shape[2:] == result.shape[2:]


if __name__ == "__main__":
    main()
