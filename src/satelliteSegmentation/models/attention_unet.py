import torch
from torch import nn
from satelliteSegmentation.config import Config
from satelliteSegmentation.models._components import Encoder, AttentionDecoder, Block


class AttentionUNet(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.encoder = Encoder(
            config.in_channels,
            list(config.hidden_channels),
            config.kernel_size,
        )

        self.bottleneck = Block(
            config.hidden_channels[-1],
            config.bottleneck_channels,
            config.kernel_size,
        )

        self.decoder = AttentionDecoder(
            config.bottleneck_channels,
            list(reversed(config.hidden_channels)),
            config.kernel_size,
        )

        self.classifier = nn.Conv2d(
            config.hidden_channels[0],
            config.num_classes,
            1,
        )

    def forward(self, x: torch.Tensor):
        x, skip_connections = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_connections)
        return self.classifier(x)

    def attention_maps(self) -> list[torch.Tensor]:
        """
        Mapas de atención [B, 1, H, W] del último forward, ordenados del nivel
        más profundo (menor resolución) al más superficial (mayor resolución).
        """
        maps = [gate.last_attention for gate in self.decoder.gates]

        if any(m is None for m in maps):
            raise RuntimeError(
                "Debes ejecutar un forward antes de consultar los mapas de atención."
            )

        return maps  # type: ignore


def main():
    conf = Config(device="cpu")
    model = AttentionUNet(conf)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parametros: {n_params:,}")

    for size in (120, 128):
        print(f"\nPrueba para {size}x{size}")
        x = torch.randn([4, 3, size, size])
        with torch.no_grad():
            result = model(x)
        print(f"Imagen entrada: {x.shape}")
        print(f"Imagen salida: {result.shape}")
        assert result.shape == (4, conf.num_classes, size, size)

        maps = model.attention_maps()
        assert len(maps) == len(conf.hidden_channels)
        for m in maps:
            assert m.shape[0] == 4 and m.shape[1] == 1
            assert 0.0 <= m.min().item() and m.max().item() <= 1.0
        print("Mapas de atencion:", [tuple(m.shape[2:]) for m in maps])


if __name__ == "__main__":
    main()
