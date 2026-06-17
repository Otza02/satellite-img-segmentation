import torch
from torch import Tensor

COLOR2LABEL = {
    (250, 235, 185): "Informal Settlements",
    (200, 200, 200): "Built-Up",
    (100, 100, 150): "Impervious Surfaces",
    (80, 140, 50): "Vegetation",
    (200, 160, 40): "Barren",
    (40, 120, 240): "Water",
    (0, 0, 0): "Unlabelled",
}
IDX2COLOR = {
    0: (250, 235, 185),
    1: (200, 200, 200),
    2: (100, 100, 150),
    3: (80, 140, 50),
    4: (200, 160, 40),
    5: (40, 120, 240),
    6: (0, 0, 0),
}
COLOR2IDX = {col: idx for idx, col in IDX2COLOR.items()}
IDX2LABEL = {idx: COLOR2LABEL[col] for idx, col in IDX2COLOR.items()}


class Tokenizer:
    @staticmethod
    def color2id(mask: Tensor) -> Tensor:
        h, w = mask.shape[1:]
        ids = torch.empty((h, w), dtype=torch.float, device=mask.device)

        for col, idx in COLOR2IDX.items():
            color = torch.tensor(col, dtype=mask.dtype, device=mask.device).view(3, 1, 1)

            matches = (mask == color).all(dim=0)
            ids[matches] = idx

        return ids

    @staticmethod
    def id2color(mask: Tensor) -> Tensor:
        mask = mask.squeeze(0)
        h, w = mask.shape
        colors = torch.zeros((3, h, w), dtype=torch.uint8, device=mask.device)

        for idx, color in IDX2COLOR.items():
            matches = mask == idx
            for c in range(3):
                colors[c][matches] = color[c]

        return colors

    @staticmethod
    def id2name(id: int) -> str:
        return IDX2LABEL[id]


def main():
    from PIL import Image
    import numpy as np
    from matplotlib import pyplot as plt

    mask = torch.from_numpy(
        np.array(Image.open("data/processed/test/masks/tile_5.37_17.png"))
    ).permute(2, 0, 1)

    mask_idx = Tokenizer.color2id(mask)
    print(torch.unique(mask_idx), mask_idx.shape)
    mask_og = Tokenizer.id2color(mask_idx)
    print(mask_og.shape)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for i, msk in enumerate([mask.permute(1, 2, 0), mask_idx, mask_og.permute(1, 2, 0)]):
        ax[i].imshow(msk)

    plt.show()


if __name__ == "__main__":
    main()
