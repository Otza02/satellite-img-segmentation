from torch.utils.data.dataset import Dataset
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
import torchvision.transforms as TF
from torchvision.transforms import v2
from torchvision import tv_tensors

from PIL import Image


class SatelliteData(Dataset):
    def __init__(
        self,
        data_dir: str | Path = "data/train",
        transform: v2.Compose | None = None,
    ):
        self.tf = transform
        data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        images_dir = data_dir / "images"
        masks_dir = data_dir / "masks"
        to_tensor = TF.ToTensor()
        X = []
        Y = []
        for file_name in tqdm(images_dir.glob("*.tif")):
            img = to_tensor(Image.open(file_name))
            X.append(img)

            msk = decode_image(masks_dir / f"{file_name.stem}.png").squeeze(0)  # type: ignore
            Y.append(msk)

        self.X = torch.stack(X)
        self.Y = torch.stack(Y)
        print(f"Dataset cargado:")
        print(f"X shape = {self.X.shape}")
        print(f"Y shape = {self.Y.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx) -> tuple[tv_tensors.Image, tv_tensors.Mask]:
        img, msk = tv_tensors.Image(self.X[idx]), tv_tensors.Mask(self.Y[idx])
        if self.tf is not None:
            return self.tf(img, msk)
        return img, msk


def main():
    from matplotlib import pyplot as plt
    from satelliteSegmentation.tokenizer import Tokenizer

    tf = v2.Compose(
        [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation([0, 180], fill=6),
        ]
    )

    data = SatelliteData(transform=tf)
    x, y = data[0]

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x.permute(1, 2, 0))
    ax[1].imshow(Tokenizer.id2color(y).permute(1, 2, 0))
    plt.show()


if __name__ == "__main__":
    main()
